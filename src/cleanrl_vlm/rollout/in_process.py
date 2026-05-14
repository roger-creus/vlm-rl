"""In-process HF generation path for COT rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CotRolloutStep:
    actions: torch.Tensor
    full_ids: torch.Tensor
    logprob_sum: torch.Tensor
    prompt_lens: torch.Tensor
    raw_texts: list[str]
    gen_truncated: torch.Tensor


def generated_span_mask(
    full_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Mask over the per-token logprob tensor ``log_probs[:, :S-1]``.

    Positions ``j ∈ [L_i - 1, S-2]`` predicting non-pad target tokens
    ``full_ids[:, j+1] != pad_id`` constitute the generated span. Rollout
    (``generate_cot_actions``) and the PPO update's re-score path both
    apply this mask; sharing it keeps ``ratio = 1`` at first-minibatch /
    first-epoch (modulo fp16 reduction-order noise).
    """
    S = full_ids.shape[1]
    positions = torch.arange(S - 1, device=full_ids.device)
    start_mask = positions.unsqueeze(0) >= (prompt_lens - 1).unsqueeze(1)
    nonpad_mask = full_ids[:, 1:] != pad_id
    mask = start_mask & nonpad_mask
    return mask.to(dtype) if dtype is not None else mask


def generate_cot_actions(
    ac_model,
    obs_batch: torch.Tensor,
    prompt_texts: list[str],
    action_names: list[str],
    max_new_tokens: int,
) -> CotRolloutStep:
    """Run one actor-adapter ``generate`` + forward; parse actions; assemble step."""
    import numpy as np

    from cleanrl_vlm.prompts.parser import parse_action_cot

    log_probs, full_ids, prompt_lens, generated_texts = ac_model.get_action(
        obs=obs_batch,
        text_prompts=prompt_texts,
    )

    B, S = full_ids.shape
    device = full_ids.device
    pad_id = ac_model.vlm.processor.tokenizer.pad_token_id
    eos_id = ac_model.vlm.processor.tokenizer.eos_token_id

    actions_list: list[int] = []
    for txt in generated_texts:
        parsed = parse_action_cot(txt, action_names)
        if parsed is None:
            parsed = int(np.random.randint(0, len(action_names)))
        actions_list.append(parsed)
    actions = torch.tensor(actions_list, dtype=torch.long, device=device)

    # Generation is truncated iff no EOS id appeared after the prompt.
    positions = torch.arange(S, device=device).unsqueeze(0)
    gen_mask = positions >= prompt_lens.unsqueeze(1)
    if eos_id is None:
        gen_truncated = torch.ones(B, dtype=torch.bool, device=device)
    else:
        gen_truncated = ((full_ids == eos_id) & gen_mask).sum(dim=1) == 0

    span_mask = generated_span_mask(full_ids, prompt_lens, pad_id, dtype=log_probs.dtype)
    logprob_sum = (log_probs * span_mask).sum(dim=-1).float()

    return CotRolloutStep(
        actions=actions,
        full_ids=full_ids,
        logprob_sum=logprob_sum,
        prompt_lens=prompt_lens,
        raw_texts=list(generated_texts),
        gen_truncated=gen_truncated,
    )
