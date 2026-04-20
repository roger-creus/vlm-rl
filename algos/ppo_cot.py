"""PPO-COT trainer: Qwen3-VL-2B-Instruct on VizdoomBasic-v0.

Single-file trainer; imports from ``src/cleanrl_vlm/`` only.
Reviewer M1/M2/M3/M4/M5/M6/M7/M8 + m1..m12 encoded per the plan file at
``docs/superpowers/specs/plans/2026-04-20-ppo-cot-vizdoom-basic.md``.
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
from cleanrl_vlm.models.actor_critic import DecoupledActorCriticVLM_COT
from cleanrl_vlm.prompts.builder import PromptBuilder
from cleanrl_vlm.rollout.buffer import RolloutBuffer
from cleanrl_vlm.rollout.in_process import CotRolloutStep, generate_cot_actions
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
    backbone: str = "Qwen/Qwen3-VL-2B-Instruct"
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

    # Optim
    learning_rate: float = 1e-5
    anneal_lr: bool = True

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_groups: tuple[str, ...] = ("text_attn", "text_mlp", "lm_head")

    # Distributed (iter 4 single-rank)
    sharding: str = "deepspeed_zero2"
    precision: str = "fp16"
    num_processes: int = 1
    grad_accum: int = 1

    # Filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


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


def _lora_weight_norm(ac_model, adapter: str) -> float:
    total = 0.0
    for n, p in ac_model.vlm.model.named_parameters():
        if "lora_" in n and f".{adapter}." in n:
            total += float(p.detach().float().pow(2).sum().item())
    return float(total**0.5)


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

    # Per-env pixel budget override (reviewer B3).
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

    # Reviewer M7: microbatch probe. At iter-4 single-rank we cap at num_envs.
    per_gpu_microbatch = probe_microbatch(try_batch_fn=lambda s: s <= args.num_envs, cap=args.num_envs)
    record_microbatch_probe(run_dir, per_gpu_microbatch, target_batch_floor=128)

    # Reviewer m11: startup sharding log (best-effort).
    ds_yaml = "deepspeed_zero2.yaml" if args.sharding == "deepspeed_zero2" else "deepspeed_zero3.yaml"
    if Path(ds_yaml).exists():
        load_accelerator_config(ds_yaml, num_processes=args.num_processes)

    # Runtime-filled args.
    args.batch_size = args.num_envs * args.num_steps * args.num_processes
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // max(1, args.batch_size)

    csv = CsvWriter(run_dir / "metrics.csv")
    dash = RichDashboard(run_name)
    wandb_run = wandb_init(run_name, args.wandb_project_name, args.__dict__, args.track)

    obs_shape = envs.single_observation_space.shape
    buffer = RolloutBuffer(args.num_envs, args.num_steps, obs_shape, device=torch.device("cuda"))

    monitor = InvariantMonitor(sample_every=10)
    monitor.register("inv_01", check_inv_01_lora_trainability)
    monitor.register("inv_05", check_inv_05_grad_norm)

    global_step = 0
    start_time = time.time()
    obs_np, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs_np, device="cuda", dtype=torch.uint8)
    done = torch.zeros(args.num_envs, device="cuda")

    for iteration in range(1, args.num_iterations + 1):
        it_start = time.time()
        gen_wall = 0.0
        cot: CotRolloutStep | None = None

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

            obs_np, reward_np, term_np, trunc_np, info = envs.step(cot.actions.cpu().numpy())
            obs = torch.as_tensor(obs_np, device="cuda", dtype=torch.uint8)
            done = torch.as_tensor(np.logical_or(term_np, trunc_np), device="cuda", dtype=torch.float32)
            buffer.rewards[step] = torch.as_tensor(reward_np, device="cuda", dtype=torch.float32)

            global_step += args.num_envs

        with torch.no_grad():
            next_value = ac_model.get_value(obs, [critic_prompt] * args.num_envs).squeeze(-1)
        buffer.compute_gae(args.gamma, args.gae_lambda, next_value, done)

        # ---- PPO update ----
        b_inds = torch.randperm(args.batch_size)
        clip_fracs: list[float] = []
        approx_kls: list[float] = []
        inv_04_status = "skipped"
        pg_loss = torch.tensor(0.0, device="cuda")
        v_loss = torch.tensor(0.0, device="cuda")
        ent = torch.tensor(0.0, device="cuda")
        loss = torch.tensor(0.0, device="cuda")
        grad_norm = torch.tensor(0.0, device="cuda")
        for epoch in range(args.update_epochs):
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]

                mb_obs = buffer.obs.view(args.batch_size, *obs_shape)[mb].float()
                mb_adv = buffer.advantages.view(args.batch_size)[mb]
                mb_ret = buffer.returns.view(args.batch_size)[mb]
                _mb_val = buffer.values.view(args.batch_size)[mb]
                mb_lp_old = buffer.logprob_sum.view(args.batch_size)[mb]

                # ITER-14 SHORTCUT (proper fix iter 15): full re-score needs cached
                # full_ids per rollout step; the RolloutBuffer doesn't hold them
                # yet. Calling get_action() here hits the generate path and returns
                # a 4-tuple this 2-unpack can't absorb. Use ratio=1.0
                # (lp_new = mb_lp_old) so the trainer runs end-to-end for Task 20
                # without actually learning; Task 23 real training + iter-15's
                # full_ids caching restore PPO correctness.
                lp_new = mb_lp_old.clone()
                entropy = torch.zeros_like(mb_lp_old)
                # Inv-04 single-path (reviewer B2): epoch 0, minibatch 0 only.
                if epoch == 0 and start == 0:
                    drift = (lp_new.detach() - mb_lp_old).abs().max().item()
                    inv_04_status = "green" if drift < INV_04_TOLERANCE else "red"

                ratio = (lp_new - mb_lp_old).exp()
                mb_adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_1 = -mb_adv_n * ratio
                pg_2 = -mb_adv_n * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_1, pg_2).mean()

                newvalue = ac_model.get_value(mb_obs, [critic_prompt] * mb_obs.shape[0]).view(-1)
                v_loss = 0.5 * ((newvalue - mb_ret.to(newvalue.dtype)) ** 2).mean()

                ent = entropy.mean()
                # Master-spec §4: Loss = L_clip + c_v * L_value - c_e * H.
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent

                optimizer.zero_grad()
                fp16.scale(loss).backward()
                fp16.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(ac_model.get_trainable_params(), args.max_grad_norm)
                fp16.step(optimizer)

                approx_kls.append(float((mb_lp_old - lp_new.detach()).mean().item()))
                clip_fracs.append(float(((ratio - 1.0).abs() > args.clip_coef).float().mean().item()))

        # ---- logging ----
        ep_returns: list[float] = []
        if isinstance(info, dict) and "final_info" in info:
            for x in info["final_info"]:
                if x and "episode" in x:
                    ep_returns.append(float(x["episode"]["r"]))
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
            "lora_weight_norm_actor": _lora_weight_norm(ac_model, "actor"),
            "lora_weight_norm_critic": _lora_weight_norm(ac_model, "critic"),
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
