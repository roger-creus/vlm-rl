"""Decoupled actor-critic VLM with dual LoRA adapters (COT interface)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.distributions.categorical import Categorical

from cleanrl_vlm.models.base_vlm import BaseVLM
from cleanrl_vlm.models.heads import CriticHead
from cleanrl_vlm.models.lora_topology import default_target_modules

AdapterName = Literal["actor", "critic"]
ACTOR: AdapterName = "actor"
CRITIC: AdapterName = "critic"


@contextmanager
def active_adapter(ac_model: DecoupledActorCriticVLM_COT, name: AdapterName) -> Iterator[None]:
    """Tripwire ctxmgr: asserts the PEFT active adapter equals ``name`` on enter.

    Setting the adapter happens inside :meth:`get_action` / :meth:`get_value`;
    this ctxmgr does NOT set, so a missing ``set_adapter`` call surfaces as an
    assertion failure rather than a silent cross-adapter forward (Inv-3).
    """
    current = _active_adapter_name(ac_model.vlm.model)
    if current != name:
        raise AssertionError(f"active_adapter expected {name!r}, got {current!r}")
    try:
        yield
    finally:
        still = _active_adapter_name(ac_model.vlm.model)
        if still != name:
            raise AssertionError(f"active_adapter mutated inside ctxmgr: {name!r} -> {still!r}")


def _active_adapter_name(peft_model) -> str:
    # PEFT's .active_adapter is a str in some versions, a single-element list in others.
    active = peft_model.active_adapter
    if isinstance(active, list):
        assert len(active) == 1, f"expected single active adapter, got {active!r}"
        return active[0]
    return active


class DecoupledActorCriticVLM_COT(nn.Module):
    """One VLM, two LoRA adapters named ``actor`` and ``critic``, plus a ``CriticHead``.

    The critic forward path uses ``.forward()`` only — it NEVER invokes
    ``.generate()``. The actor ``.generate()`` path is the only generation path.
    """

    def __init__(
        self,
        vlm_name: str,
        min_pixels: int,
        max_pixels: int,
        attn_implementation: str,
        dtype: torch.dtype,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_groups: tuple[str, ...],
        max_new_tokens: int,
    ) -> None:
        super().__init__()
        self.vlm = BaseVLM(
            vlm_name=vlm_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        hidden_size = self.vlm.model.config.text_config.hidden_size
        # Critic head stays fp32 so GradScaler can unscale its grads. Forward
        # output is cast to the VLM's dtype at the call-site in get_value() if
        # downstream arithmetic requires it.
        self.critic_head = CriticHead(hidden_size).to(device=self.vlm.model.device)
        self.max_new_tokens = max_new_tokens

        # One LoraConfig shared across both adapters so target_modules cannot
        # diverge between actor and critic (PEFT consumes the list in-order).
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(default_target_modules(set(lora_groups))),
            bias="none",
        )
        self.vlm.model = get_peft_model(self.vlm.model, cfg, adapter_name=ACTOR)
        self.vlm.model.add_adapter(CRITIC, cfg)

        # Re-freeze: only lora_* params train (critic head handled separately in
        # get_trainable_params). Cast trainable LoRA params to fp32 so the
        # GradScaler can unscale their gradients (GradScaler refuses to unscale
        # fp16 grads — "Attempting to unscale FP16 gradients").
        for n, p in self.vlm.model.named_parameters():
            trainable = "lora_" in n
            p.requires_grad = trainable
            if trainable:
                p.data = p.data.float()

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        params = self.vlm.get_trainable_params()
        params.extend(list(self.critic_head.parameters()))
        return params

    def actor_param_ids(self) -> set[int]:
        return _adapter_param_ids(self.vlm.model, ACTOR)

    def critic_param_ids(self) -> set[int]:
        ids = _adapter_param_ids(self.vlm.model, CRITIC)
        ids |= {id(p) for p in self.critic_head.parameters() if p.requires_grad}
        return ids

    def get_action(
        self,
        obs: torch.Tensor,
        text_prompts: list[str],
        action_ids: torch.Tensor | None = None,
        prompt_lens: torch.Tensor | None = None,
    ):
        self.vlm.model.set_adapter(ACTOR)
        with active_adapter(self, ACTOR):
            inputs = self.vlm.preprocess_obs_and_text(obs, text_prompts, add_generation_prompt=True)
            batch_size = len(text_prompts)
            if action_ids is None:
                full_ids = self.vlm.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                )
                generated_texts = self.vlm.processor.batch_decode(
                    full_ids[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
                prompt_lens = torch.tensor(
                    [inputs.input_ids.shape[1]] * batch_size,
                    device=self.vlm.model.device,
                )
            else:
                full_ids = action_ids
            attention_mask = (full_ids != self.vlm.processor.tokenizer.pad_token_id).long()
            # Extend mm_token_type_ids with zeros for generated positions so Qwen3-VL's
            # M-RoPE can compute position ids over the full span. Original
            # mm_token_type_ids has shape [B, prompt_len]; pad to [B, full_ids.shape[1]].
            mm_tti = getattr(inputs, "mm_token_type_ids", None)
            if mm_tti is not None:
                pad_len = full_ids.shape[1] - mm_tti.shape[1]
                if pad_len > 0:
                    mm_tti = torch.cat(
                        [
                            mm_tti,
                            torch.zeros(
                                mm_tti.shape[0],
                                pad_len,
                                dtype=mm_tti.dtype,
                                device=mm_tti.device,
                            ),
                        ],
                        dim=1,
                    )
            outputs = self.vlm.model(
                input_ids=full_ids,
                image_grid_thw=inputs.image_grid_thw,
                pixel_values=inputs.pixel_values,
                mm_token_type_ids=mm_tti,
                output_hidden_states=True,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            log_probs_all = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = full_ids[:, 1:]
            log_probs = torch.gather(log_probs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            entropy = Categorical(logits=logits[:, :-1, :]).entropy()
            if action_ids is None:
                return log_probs, full_ids, prompt_lens, generated_texts
            return log_probs, entropy

    def get_value(self, obs: torch.Tensor, prompt_text: list[str]) -> torch.Tensor:
        """Critic path: ``.forward()`` only, never ``.generate()``."""
        self.vlm.model.set_adapter(CRITIC)
        with active_adapter(self, CRITIC):
            inputs = self.vlm.preprocess_obs_and_text(obs, prompt_text)
            outputs = self.vlm.model(**inputs, output_hidden_states=True)
            last_hidden = self.vlm.last_hidden_state(
                outputs.hidden_states[-1],
                inputs["attention_mask"],
            )
            # critic_head is fp32 to keep GradScaler happy; cast the VLM hidden
            # (fp16 in default precision) before feeding in.
            return self.critic_head(last_hidden.float())


def _adapter_param_ids(peft_model, adapter: AdapterName) -> set[int]:
    # NOTE: we do NOT filter by ``p.requires_grad`` here — PEFT's
    # ``set_adapter(name)`` flips requires_grad on the non-active adapter,
    # which would make disjointness assertions depend on current state.
    # The identity set is about parameter ids, not mutable flags.
    tag = f".{adapter}."
    return {id(p) for n, p in peft_model.named_parameters() if "lora_" in n and tag in n}


def lora_params_for(peft_model, adapter: AdapterName | None = None) -> list[torch.nn.Parameter]:
    """Return the list of ``lora_*`` parameters, optionally filtered to one adapter.

    Cached lookups let the trainer avoid re-scanning ``named_parameters()``
    per minibatch (e.g., when re-enabling ``requires_grad`` after PEFT's
    ``set_adapter`` freezes the non-active side).
    """
    tag = None if adapter is None else f".{adapter}."
    return [p for n, p in peft_model.named_parameters() if "lora_" in n and (tag is None or tag in n)]
