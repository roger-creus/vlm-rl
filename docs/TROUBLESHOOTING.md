# Troubleshooting

Collected failure modes and diagnostic paths. Populated by /loop as issues surface.

## Install

- `flash-attn` build fails with CUDA mismatch → ensure `nvcc --version` matches the CUDA PyTorch was built against. Use `TORCH_CUDA_ARCH_LIST` env var to limit arches. Falls back to SDPA at runtime, but a build failure breaks full install.

## Runtime (populated as /loop encounters issues)

- OOM on backbone load → reduce `min_pixels`/`max_pixels` in the processor config; enable gradient checkpointing.
- Logprob drift between vLLM and HF forward (Inv-4) → check tokenizer mismatch, attention mask divergence, stale adapter on vLLM server, image preprocessing path difference.
- NaN in gradients (Inv-6) → check loss scale history; reduce LR; dump the offending microbatch from `runs/<name>/nan_dumps/`.
