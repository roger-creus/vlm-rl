# Algorithms

Placeholder. A per-algorithm page ships with each new `algos/*.py` file per master-spec S-8.

## Planned canon (master-spec §5)

| file | interface | algorithm | status |
|------|-----------|-----------|--------|
| `algos/ppo_cot.py`    | COT            | PPO (clipped surrogate + GAE) | not started |
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
