# Contributing

Master-spec §11 defines 12 standing rules (S-1..S-12) that govern every PR. Quick reference:

- **S-1** — Correctness coverage never decreases. New file under `algos/`, `experimental/`, or `envs/` ships with matching tests in the same PR.
- **S-3** — Perf optimizations prove Inv-4 parity before merge.
- **S-5** — External-library changes (vllm, accelerate, deepspeed, peft, transformers, flash_attn) require a web-fetch of current docs with summary in the PR body.
- **S-6, S-7, S-8** — Backbone / env / algorithm onboarding rituals.
- **S-12** — Every PR updates `CHANGELOG.md` with { what, why, evidence, invariants-run }.

See the master spec for the full list.
