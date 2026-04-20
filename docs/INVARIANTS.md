# Correctness invariants

Master-spec §8 canonical list. Each invariant = a pytest test in `tests/invariants/test_inv_<NN>.py` + a runtime `InvariantMonitor` hook.

Operating principle (master-spec §0): thresholds are signals, not CI cliffs. Hard-fail applies only to genuine correctness bugs.

| id | what it checks | binary / tunable | location |
|----|----------------|------------------|----------|
| Inv-1  | LoRA trainability split (lora_/head params train; base frozen)                          | binary   | `tests/invariants/test_inv_01.py` |
| Inv-2  | LoRA weights actually change over training                                               | tunable  | `tests/invariants/test_inv_02.py` |
| Inv-3  | Active adapter sanity (actor/critic set correctly per forward)                           | binary   | `tests/invariants/test_inv_03.py` |
| Inv-4  | Training ↔ inference logprob parity (vLLM vs HF forward)                                 | tunable  | `tests/invariants/test_inv_04.py` |
| Inv-5  | Gradient-norm correctness cross-check                                                     | binary   | `tests/invariants/test_inv_05.py` |
| Inv-6  | FP16 stability (GradScaler + NaN/Inf watch)                                              | tunable  | `tests/invariants/test_inv_06.py` |
| Inv-7  | Checkpoint round-trip (save → load → deterministic rollout matches)                       | binary   | `tests/invariants/test_inv_07.py` |
| Inv-8  | Image-input correctness (patch coverage, resolution floor, cross-rank token count)         | binary   | `tests/invariants/test_inv_08.py` |
| Inv-9  | Reward-pipeline integrity (scripted rewards survive to advantage)                          | binary   | `tests/invariants/test_inv_09.py` |
| Inv-10 | Episode-boundary masking (GAE resets at `done=True`)                                      | binary   | `tests/invariants/test_inv_10.py` |
| Inv-11 | Determinism under fixed seed (bitwise-identical re-runs)                                  | binary   | `tests/invariants/test_inv_11.py` |
| Inv-12 | Resume parity (continuous 200 ≈ 100 + resume + 100)                                       | tunable  | `tests/invariants/test_inv_12.py` |
| Inv-13 | Padding & image-token masking (zero gradient contribution)                                | binary   | `tests/invariants/test_inv_13.py` |
| Inv-14 | Distributed-broadcast agreement (LoRA weights identical across ranks)                      | binary   | `tests/invariants/test_inv_14.py` |
| Inv-15 | Ground-truth vision probe (VLM actually perceives the scene)                              | tunable  | `scripts/probe_vision.py` + `tests/invariants/test_inv_15.py` |

Test files populated by /loop as the corresponding infrastructure lands.
