# Troubleshooting

Collected failure modes and diagnostic paths. Populated by /loop as issues surface.

## Install

- `flash-attn` build fails with `OSError: CUDA_HOME environment variable is not set` → point it at a CUDA toolkit ≥ the one PyTorch was built against. On the Mila cluster (torch 2.6.0+cu124): `export CUDA_HOME=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/12.5.0` and prepend `$CUDA_HOME/bin` to `PATH` before `uv pip install flash-attn --no-build-isolation`. Same trick for any cluster that ships CUDA under a non-default path.
- `flash-attn` build fails with CUDA mismatch → ensure `nvcc --version` matches the CUDA PyTorch was built against. Use `TORCH_CUDA_ARCH_LIST` env var to limit arches. Falls back to SDPA at runtime, but a build failure breaks full install.
- `uv sync` fails with "Dependency #N cannot be a direct reference unless `tool.hatch.metadata.allow-direct-references` is true" → already set in `pyproject.toml` (required because `transformers @ git+https://...` is a direct reference). If it resurfaces, double-check the config wasn't dropped by a merge.

## Runtime (populated as /loop encounters issues)

- OOM on backbone load → reduce `min_pixels`/`max_pixels` in the processor config; enable gradient checkpointing.
- Logprob drift between vLLM and HF forward (Inv-4) → check tokenizer mismatch, attention mask divergence, stale adapter on vLLM server, image preprocessing path difference.
- NaN in gradients (Inv-6) → check loss scale history; reduce LR; dump the offending microbatch from `runs/<name>/nan_dumps/`.
