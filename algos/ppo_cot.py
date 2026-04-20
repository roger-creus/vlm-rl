"""PPO-COT trainer: Qwen3-VL-2B-Instruct on VizdoomBasic-v1.

Single-file trainer; imports from ``src/cleanrl_vlm/`` only.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
from gymnasium.vector import AsyncVectorEnv

from cleanrl_vlm.envs.registry import make_env
from cleanrl_vlm.envs.vizdoom.action_tables import action_tables
from cleanrl_vlm.models.actor_critic import (
    ACTOR,
    CRITIC,
    DecoupledActorCriticVLM_COT,
    lora_params_for,
)
from cleanrl_vlm.prompts.builder import PromptBuilder
from cleanrl_vlm.rollout.buffer import RolloutBuffer
from cleanrl_vlm.rollout.in_process import CotRolloutStep, generate_cot_actions, generated_span_mask
from cleanrl_vlm.training.checkpoint import save_vlm_actor_critic_checkpoint
from cleanrl_vlm.training.distributed import load_accelerator_config
from cleanrl_vlm.training.invariants import (
    INV_04_TOLERANCE,
    InvariantMonitor,
    check_inv_01_lora_trainability,
    check_inv_05_grad_norm,
)
from cleanrl_vlm.training.logging import CsvWriter, RichDashboard, wandb_init
from cleanrl_vlm.training.microbatch_probe import probe_microbatch, record_microbatch_probe
from cleanrl_vlm.training.precision import Fp16State

ALGO_SLUG = "ppo_cot"


@dataclass
class Args:
    # Run meta
    exp_name: str = "ppo_cot"
    seed: int = 0
    track: bool = False
    wandb_project_name: str = "cleanRL-VLM"
    wandb_entity: str | None = None
    checkpoint_interval: int = 10

    # Env
    env_id: str = "VizdoomBasic-v1"
    env_config: str = "configs/envs/VizdoomBasic-v1.yaml"
    num_envs: int = 4

    # Backbone
    backbone: str = "Qwen/Qwen3.5-2B"
    backbone_config: str = "configs/backbones.yaml"
    max_new_tokens: int = 256

    # Algo
    num_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 4
    total_timesteps: int = 200_000
    # Per spec §14: reward scaling 0.01 is the default. Multiplied into
    # rewards before buffer storage so GAE / value loss use the scaled
    # signal. RecordEpisodeStatistics still emits UNSCALED returns
    # (wrapper sits inside AsyncVectorEnv → ep_return_mean is in raw env
    # units regardless of this setting).
    reward_scale: float = 0.01

    # Optim
    learning_rate: float = 1e-5
    anneal_lr: bool = True

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_groups: tuple[str, ...] = (
        "text_attn",
        "text_mlp",
        "vision_attn",
        "vision_mlp",
        "merger",
        "lm_head",
    )

    # Distributed
    sharding: str = "deepspeed_zero2"
    # BF16 default (not FP16 as master-spec §1 suggested): Qwen3.5's Gated
    # DeltaNet backward is numerically unstable in FP16 when the fast-path
    # flash-linear-attention is not installed — produces NaN gradients at
    # linear_attn.in_proj_qkv despite finite forward output. BF16's wider
    # exponent avoids the underflow path. See amendment
    # 2026-04-20-bf16-default-for-qwen3.5.md.
    precision: str = "bf16"
    num_processes: int = 1
    grad_accum: int = 1


def _build_run_name(args: Args) -> str:
    date = time.strftime("%Y-%m-%d")
    slug = args.backbone.split("/")[-1].lower()
    return f"{args.exp_name}__{args.env_id}__{slug}__{args.seed}__{date}"


def _set_seed(seed: int) -> None:
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("HF_SEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _lora_weight_norm(params: list[torch.nn.Parameter]) -> float:
    if not params:
        return 0.0
    with torch.no_grad():
        total = torch.stack([p.detach().float().pow(2).sum() for p in params]).sum()
    return float(total.sqrt().item())


def _extract_ep_returns(info) -> list[float]:
    """Pull episode returns out of a gymnasium AsyncVectorEnv info dict.

    Handles three emission patterns across gymnasium versions. The
    branches are **mutually exclusive** — gymnasium 1.x populates both
    ``info["episode"]`` and ``info["final_info"]`` on the same terminal
    step, so we pick exactly one source per call (precedence: modern
    top-level → older `final_info` list → intermediate `final_info`
    dict-of-arrays). This prevents each terminal return from being
    counted twice in `ep_return_n`.
    """
    out: list[float] = []
    if not isinstance(info, dict):
        return out
    ep = info.get("episode")
    mask = info.get("_episode")
    if isinstance(ep, dict) and "r" in ep:
        r = np.asarray(ep["r"]).reshape(-1)
        if mask is not None:
            m = np.asarray(mask, dtype=bool).reshape(-1)
            for v, b in zip(r, m, strict=False):
                if b:
                    out.append(float(v))
        else:
            out.extend(float(v) for v in r)
        return out
    fi = info.get("final_info")
    if isinstance(fi, list):
        for item in fi:
            if isinstance(item, dict) and "episode" in item:
                out.append(float(item["episode"]["r"]))
        return out
    if isinstance(fi, dict) and "episode" in fi:
        sub = fi["episode"]
        if isinstance(sub, dict) and "r" in sub:
            out.extend(float(v) for v in np.asarray(sub["r"]).reshape(-1))
    return out


def main() -> None:  # noqa: C901  (trainer orchestration; complexity expected)
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    env_cfg = yaml.safe_load(Path(args.env_config).read_text())
    backbones = yaml.safe_load(Path(args.backbone_config).read_text())
    bb = backbones[args.backbone]

    run_name = _build_run_name(args)
    run_dir = Path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    # Per-env pixel budget may override the backbone default.
    min_pixels = int(env_cfg.get("processor_min_pixels", bb["processor_pixel_budget"]["min_pixels"]))
    max_pixels = int(env_cfg.get("processor_max_pixels", bb["processor_pixel_budget"]["max_pixels"]))

    action_names = action_tables[args.env_id]
    prompt_builder = PromptBuilder(args.env_id, action_names)
    actor_prompt = prompt_builder.actor_prompt()
    critic_prompt = prompt_builder.critic_prompt()

    envs = AsyncVectorEnv([make_env(args.env_id, env_cfg, args.seed, i, run_name) for i in range(args.num_envs)])

    ac_model = DecoupledActorCriticVLM_COT(
        vlm_name=args.backbone,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        attn_implementation=bb["attn_implementation"],
        dtype=torch.float16 if args.precision == "fp16" else torch.bfloat16,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_groups=args.lora_groups,
        max_new_tokens=args.max_new_tokens,
    )
    optimizer = torch.optim.AdamW(ac_model.get_trainable_params(), lr=args.learning_rate)
    fp16 = Fp16State(enabled=(args.precision == "fp16"))

    # Conservative placeholder — returns min(num_envs, cap), not a real
    # OOM-triggering auto-probe. The master-spec §3 "probe doubling until
    # OOM" feature is backlog; for now we assume num_envs already fits.
    # The artifact is still written for record-keeping.
    per_gpu_microbatch = probe_microbatch(try_batch_fn=lambda s: s <= args.num_envs, cap=args.num_envs)
    record_microbatch_probe(run_dir, per_gpu_microbatch, target_batch_floor=128)

    ds_yaml = "deepspeed_zero2.yaml" if args.sharding == "deepspeed_zero2" else "deepspeed_zero3.yaml"
    if Path(ds_yaml).exists():
        load_accelerator_config(ds_yaml, num_processes=args.num_processes)

    # Per-process batch shape matches buffer.obs = [num_steps, num_envs, ...].
    # Include num_processes only when aggregating total env-steps for logging
    # and the timesteps budget, not when indexing per-rank buffers.
    batch_size = args.num_envs * args.num_steps
    global_batch_size = batch_size * args.num_processes
    num_iterations = args.total_timesteps // max(1, global_batch_size)

    csv = CsvWriter(run_dir / "metrics.csv")
    dash = RichDashboard(run_name)
    wandb_run = wandb_init(run_name, args.wandb_project_name, args.__dict__, args.track)

    obs_shape = envs.single_observation_space.shape
    buffer = RolloutBuffer(args.num_envs, args.num_steps, obs_shape, device=torch.device("cuda"))

    lora_params_all = lora_params_for(ac_model.vlm.model)
    lora_params_actor = lora_params_for(ac_model.vlm.model, ACTOR)
    lora_params_critic = lora_params_for(ac_model.vlm.model, CRITIC)

    monitor = InvariantMonitor(sample_every=10)
    monitor.register("inv_01", check_inv_01_lora_trainability)
    monitor.register("inv_05", check_inv_05_grad_norm)

    pad_id = ac_model.vlm.processor.tokenizer.pad_token_id

    global_step = 0
    start_time = time.time()
    obs_np, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs_np, device="cuda", dtype=torch.uint8)
    done = torch.zeros(args.num_envs, device="cuda")

    for iteration in range(1, num_iterations + 1):
        it_start = time.time()
        gen_wall = 0.0
        cot: CotRolloutStep | None = None
        ep_returns: list[float] = []

        # ---- rollout ----
        for step in range(args.num_steps):
            buffer.obs[step] = obs
            buffer.dones[step] = done

            with torch.no_grad():
                t0 = time.time()
                cot = generate_cot_actions(
                    ac_model,
                    obs,
                    [actor_prompt] * args.num_envs,
                    action_names,
                    args.max_new_tokens,
                )
                gen_wall += time.time() - t0
                value = ac_model.get_value(obs, [critic_prompt] * args.num_envs).squeeze(-1)

            buffer.actions[step] = cot.actions
            buffer.logprob_sum[step] = cot.logprob_sum
            buffer.values[step] = value
            # Cache full_ids + prompt_lens per step so the PPO update can re-score
            # under the (now updated) actor adapter without re-running generate.
            buffer.full_ids_per_step[step] = cot.full_ids.detach()
            buffer.prompt_lens_per_step[step] = cot.prompt_lens.detach()

            obs_np, reward_np, term_np, trunc_np, info = envs.step(cot.actions.cpu().numpy())
            obs = torch.as_tensor(obs_np, device="cuda", dtype=torch.uint8)
            done = torch.as_tensor(np.logical_or(term_np, trunc_np), device="cuda", dtype=torch.float32)
            # Apply spec §14 reward_scale (0.01 default) to tame value-loss
            # magnitudes and keep FP16 GradScaler stable.
            buffer.rewards[step] = torch.as_tensor(reward_np * args.reward_scale, device="cuda", dtype=torch.float32)
            ep_returns.extend(_extract_ep_returns(info))

            global_step += args.num_envs

        with torch.no_grad():
            next_value = ac_model.get_value(obs, [critic_prompt] * args.num_envs).squeeze(-1)
        buffer.compute_gae(args.gamma, args.gae_lambda, next_value, done)

        # ---- PPO update ----
        # Inv-4 bit-parity requires the re-score forward to run at the
        # *same batch size* and *same padding pattern* as the rollout's
        # forward. Rollout runs each step at batch=num_envs. Batching
        # rows across steps (with variable S_t) introduces padding +
        # batch-size divergence that shifts fp16 output by ~1e-2 for
        # padded rows. We therefore shuffle at **step** granularity and
        # re-score each step intact at batch=num_envs — same shape as
        # rollout, so flash-attn dispatches the same kernel and the
        # output matches bit-for-bit. A single minibatch bundles
        # ``steps_per_mb = num_steps // num_minibatches`` consecutive
        # rollout steps so loss aggregation still sees ``minibatch_size``
        # rows at a time.
        assert args.num_steps % args.num_minibatches == 0, (
            f"num_steps ({args.num_steps}) must be divisible by num_minibatches "
            f"({args.num_minibatches}) for step-grouped re-score (Inv-4 parity)."
        )
        steps_per_mb = args.num_steps // args.num_minibatches

        step_inds = torch.randperm(args.num_steps, device="cuda").cpu().tolist()
        clip_fracs: list[float] = []
        approx_kls: list[float] = []
        inv_04_status = "skipped"
        pg_loss = torch.tensor(0.0, device="cuda")
        v_loss = torch.tensor(0.0, device="cuda")
        ent = torch.tensor(0.0, device="cuda")
        loss = torch.tensor(0.0, device="cuda")
        grad_norm = torch.tensor(0.0, device="cuda")
        for epoch in range(args.update_epochs):
            for mb_start in range(0, args.num_steps, steps_per_mb):
                mb_steps = step_inds[mb_start : mb_start + steps_per_mb]

                # Re-score each step at batch=num_envs (matches rollout's
                # forward shape for both actor and critic — keeps flash-attn
                # kernel dispatch identical so Inv-4 parity holds on the
                # actor side AND value targets don't silently drift from the
                # cached rollout values on the critic side).
                lp_new_parts: list[torch.Tensor] = []
                entropy_parts: list[torch.Tensor] = []
                newvalue_parts: list[torch.Tensor] = []
                for t in mb_steps:
                    step_full_ids = buffer.full_ids_per_step[t]
                    step_prompt_lens = buffer.prompt_lens_per_step[t]
                    assert step_full_ids is not None and step_prompt_lens is not None
                    step_obs = buffer.obs[t].float()  # [num_envs, H, W, C]
                    log_probs, ent_t = ac_model.get_action(
                        obs=step_obs,
                        text_prompts=[actor_prompt] * args.num_envs,
                        action_ids=step_full_ids,
                        prompt_lens=step_prompt_lens,
                    )
                    span_mask = generated_span_mask(step_full_ids, step_prompt_lens, pad_id, dtype=log_probs.dtype)
                    span_count = span_mask.sum(dim=-1).clamp(min=1)
                    lp_new_parts.append((log_probs * span_mask).sum(dim=-1))
                    entropy_parts.append((ent_t * span_mask).sum(dim=-1) / span_count)
                    newvalue_parts.append(ac_model.get_value(step_obs, [critic_prompt] * args.num_envs).view(-1))
                lp_new = torch.cat(lp_new_parts)
                entropy = torch.cat(entropy_parts)
                newvalue = torch.cat(newvalue_parts)

                # Flat [batch_size] layout mirrors buffer.obs.view(batch_size, ...)
                # ordering (step-major, env-minor) so we can gather matching
                # advantages / returns / stored logprob_sum.
                mb_indices = torch.tensor(
                    [t * args.num_envs + b for t in mb_steps for b in range(args.num_envs)],
                    device="cuda",
                    dtype=torch.long,
                )
                mb_adv = buffer.advantages.view(batch_size)[mb_indices]
                mb_ret = buffer.returns.view(batch_size)[mb_indices]
                mb_lp_old = buffer.logprob_sum.view(batch_size)[mb_indices]

                # Inv-04 single-path: first minibatch of first epoch must
                # match the rollout's cached logprob_sum (ratio ≈ 1).
                if epoch == 0 and mb_start == 0:
                    drift = (lp_new.detach() - mb_lp_old).abs().max().item()
                    inv_04_status = "green" if drift < INV_04_TOLERANCE else "red"

                ratio = (lp_new - mb_lp_old).exp()
                mb_adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_1 = -mb_adv_n * ratio
                pg_2 = -mb_adv_n * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_1, pg_2).mean()

                v_loss = 0.5 * ((newvalue - mb_ret.to(newvalue.dtype)) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent

                # PEFT's set_adapter(name) calls requires_grad_(False) on the
                # non-active adapter. After get_value switches to CRITIC, the
                # ACTOR LoRA params end up frozen and .backward() skips them.
                # Re-enable both before backward; each loss only contributes
                # gradients through the adapter it was forwarded under, so
                # correctness is preserved.
                for p in lora_params_all:
                    p.requires_grad_(True)

                optimizer.zero_grad()
                fp16.scale(loss).backward()
                fp16.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(ac_model.get_trainable_params(), args.max_grad_norm)
                fp16.step(optimizer)

                approx_kls.append(float((mb_lp_old - lp_new.detach()).mean().item()))
                clip_fracs.append(float(((ratio - 1.0).abs() > args.clip_coef).float().mean().item()))

        # ---- logging ----
        # ep_returns was accumulated during rollout per _extract_ep_returns.
        row = {
            "global_step": global_step,
            "iteration": iteration,
            "total_env_steps": global_step,
            "wall_s": time.time() - start_time,
            "loss_total": float(loss.item()),
            "loss_clip": float(pg_loss.item()),
            "loss_value": float(v_loss.item()),
            "loss_entropy": float(ent.item()),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "grad_norm_global": float(grad_norm),
            "loss_scale": fp16.current_scale(),
            "lr": args.learning_rate,
            "ep_return_mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "ep_return_n": len(ep_returns),
            "generate_wall_s": gen_wall,
            "rollout_wall_s": time.time() - it_start,
            "train_wall_s": 0.0,
            "lora_weight_norm_actor": _lora_weight_norm(lora_params_actor),
            "lora_weight_norm_critic": _lora_weight_norm(lora_params_critic),
            "adapter_sync_wall_s": 0.0,
            "gen_truncated_rate": float(cot.gen_truncated.float().mean().item()) if cot else 0.0,
            "inv_4_status": inv_04_status,
        }
        inv_results = monitor.maybe_run(
            iteration,
            {"ac_model": ac_model, "params": ac_model.get_trainable_params()},
        )
        for r in inv_results.values():
            row[f"{r.name}_status"] = r.status
        csv.log(row)
        dash.update(row)
        if wandb_run is not None:
            import wandb

            wandb.log(row, step=global_step)

        if iteration % args.checkpoint_interval == 0:
            save_vlm_actor_critic_checkpoint(
                run_dir / "checkpoints" / f"step_{global_step:06d}",
                algo_slug=ALGO_SLUG,
                ac_model=ac_model,
                optimizer=optimizer,
                rng_state={"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
                step=global_step,
                wandb_run_id=None,
                manifest={"args": args.__dict__, "run_name": run_name},
            )

    csv.close()


if __name__ == "__main__":
    main()
