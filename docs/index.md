# cleanrl-vlm

CleanRL-style library + research paper for online RL finetuning of Vision-Language Models in interactive visual environments (ViZDoom, Atari, Minigrid).

## Quickstart

See the [README](../README.md).

## Pages

- [Architecture](ARCHITECTURE.md) — library layout, rollout/train split, LoRA dual-adapter pattern.
- [Algorithms](ALGORITHMS.md) — per-algorithm math + code pointers (populated as canon trainers land).
- [Environments](ENVS.md) — env catalogue with target scores.
- [Backbones](BACKBONES.md) — supported VLMs with memory footprint notes.
- [Recipes](RECIPES.md) — copy-pasteable reproduction commands.
- [Results](RESULTS.md) — live benchmark dashboard (auto-generated).
- [Research](RESEARCH.md) — research journal.
- [Invariants](INVARIANTS.md) — correctness invariants Inv-1..Inv-15.
- [Checkpointing](CHECKPOINTING.md) — save/resume details.
- [Logging](LOGGING.md) — rich/wandb/CSV interplay + metric glossary.
- [Contributing](CONTRIBUTING.md) — onboarding rituals for new envs/algos/backbones.
- [Troubleshooting](TROUBLESHOOTING.md) — common failure modes.

## Status

At the bootstrap milestone the library is a runnable scaffold — `pytest -m tier1 -v` passes a "hello VLM" smoke test that loads Qwen3.5-VL-0.8B and generates once. The autonomous `/loop` agent takes over from here per master-spec §13.

## Master spec

The single source of truth: [`docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`](superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md).
