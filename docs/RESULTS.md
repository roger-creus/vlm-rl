# Results

Live benchmark dashboard — **auto-generated** by `scripts/build_dashboard.py` from Tier-2 run metric CSVs.

At bootstrap there are no results yet. Populated by /loop as Tier-2 runs complete.

| env | algo | interface | backbone | seeds | status | curve |
|-----|------|-----------|----------|-------|--------|-------|
| VizdoomBasic-v1 | PPO | COT | Qwen3-VL-2B-Instruct | 0 | **yellow** | smoke run iter 19 (10 iters); full campaign pending. Correctness milestones landed: LoRA actor trains (Inv-2 green), FP16 stable (Inv-6 green at 32768 scale throughout), ep_return captured, Inv-4 single-path green 9/10 iters. Learning signal inconclusive at 10-iter scale (returns 95 / -410 / -410 — noisy). |
