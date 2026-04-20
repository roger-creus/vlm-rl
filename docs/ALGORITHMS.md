# Algorithms

## PPO-COT

Chain-of-thought PPO on VLMs. The actor VLM generates free-form text under
the COT prompt; the parser extracts `ACTION: <NAME>` from the tail (last
regex match, whitelisted against the env's action names). The scalar logprob
per trajectory is the sum of token logprobs over the generated span; the PPO
ratio is sequence-level.

The critic shares the same base VLM via a second LoRA adapter; a
`CriticHead` MLP reads the last non-pad hidden state. GAE(λ) runs on env
rewards (scaled by `reward_scale=0.01` per spec §14 to keep FP16 GradScaler
stable).

Signed loss (master-spec §4):

```
Loss = mean(L_clip) + c_v * mean(L_value) - c_e * H
```

- Canon impl: `algos/ppo_cot.py`.
- Default backbone: `Qwen/Qwen3-VL-2B-Instruct` (Tier-1).
- Invariants at iter-4 scope: Inv-1, Inv-3, Inv-4 (single-path), Inv-5,
  Inv-6, Inv-9, Inv-10, Inv-11, Inv-13.

---

Placeholder. A per-algorithm page ships with each new `algos/*.py` file per master-spec S-8.

## Planned canon (master-spec §5)

| file | interface | algorithm | status |
|------|-----------|-----------|--------|
| `algos/ppo_cot.py`    | COT            | PPO (clipped surrogate + GAE) | **landed** (iter 4 v1) |
| `algos/ppo_action.py` | action-scoring | PPO                           | not started |
| `algos/ppo_head.py`   | MLP head       | PPO                           | not started |
| `algos/grpo_cot.py`    | COT            | GRPO (group-relative)          | not started |
| `algos/grpo_action.py` | action-scoring | GRPO                           | not started |
| `algos/grpo_head.py`   | MLP head       | GRPO                           | not started |
| `algos/rloo_cot.py`    | COT            | RLOO (leave-one-out)           | not started |
| `algos/rloo_action.py` | action-scoring | RLOO                           | not started |
| `algos/rloo_head.py`   | MLP head       | RLOO                           | not started |

## Baselines (master-spec §7)

| file | what | status |
|------|------|--------|
| `baselines/cnn_ppo.py`         | from-scratch CNN PPO (non-VLM)              | not started |
| `baselines/zero_shot_vlm.py`   | pure prompting, no finetuning               | not started |
| `baselines/frozen_vlm_head.py` | frozen VLM + trainable MLP head (no LoRA)   | not started |
