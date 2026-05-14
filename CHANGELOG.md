# Changelog

## Unreleased

Public handoff cleanup:

- Renamed public-facing project metadata and docs to `vlm-rl`.
- Removed stale prototype entrypoints, local planning artifacts, and the generated MkDocs site.
- Made `pyproject.toml` and `uv.lock` the canonical installation path.
- Kept the supported code surface focused on `src/cleanrl_vlm/`, `algos/ppo_cot.py`, configs, scripts, tests, and docs.

Current research state:

- PPO-COT trainer for VLM actor generation plus PPO updates.
- Dual LoRA actor/critic adapter topology.
- Qwen3.5-oriented backbone config with BF16 default.
- ViZDoom, Atari Pong, and MiniGrid environment factories.
- Prompt templates for Tier-1 environments.
- Unit and invariant test coverage for rollout buffers, GAE, action parsing, LoRA topology, logging, checkpoint structure, environment factories, and core training invariants.
