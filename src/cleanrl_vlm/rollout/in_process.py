"""In-process HF generation path for COT rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CotRolloutStep:
    """Typed return from :func:`generate_cot_actions` (reviewer M1).

    All tensors live on the model device unless otherwise noted.
    """

    actions: torch.LongTensor  # [B] parsed int action per env (uniform-sampled on parse fail)
    full_ids: torch.LongTensor  # [B, S] prompt + generated ids, padded
    logprob_sum: torch.FloatTensor  # [B] sum of token logprobs over the generated span
    prompt_lens: torch.LongTensor  # [B] prompt length per row
    raw_texts: list[str]  # len B decoded generation
    gen_truncated: torch.BoolTensor  # [B] True iff generation hit max_new_tokens w/o EOS (reviewer m4)


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

    B = full_ids.shape[0]
    actions = torch.zeros(B, dtype=torch.long, device=full_ids.device)
    gen_truncated = torch.zeros(B, dtype=torch.bool, device=full_ids.device)
    eos_id = ac_model.vlm.processor.tokenizer.eos_token_id
    for i, txt in enumerate(generated_texts):
        parsed = parse_action_cot(txt, action_names)
        if parsed is None:
            parsed = int(np.random.randint(0, len(action_names)))
        actions[i] = parsed
        # Truncated iff no EOS id appeared among the generated tail.
        gen_tail = full_ids[i, int(prompt_lens[i]) :]
        if eos_id is None or (gen_tail == eos_id).sum().item() == 0:
            gen_truncated[i] = True

    # logprob_sum over generated span: log_probs is aligned to target_ids = full_ids[:, 1:].
    # Sum positions >= prompt_len - 1 (predicting tokens from prompt_len onwards).
    logprob_sum = torch.zeros(B, dtype=torch.float32, device=full_ids.device)
    for i in range(B):
        start = int(prompt_lens[i]) - 1
        logprob_sum[i] = log_probs[i, start:].sum().float()

    return CotRolloutStep(
        actions=actions,
        full_ids=full_ids,
        logprob_sum=logprob_sum,
        prompt_lens=prompt_lens,
        raw_texts=list(generated_texts),
        gen_truncated=gen_truncated,
    )
