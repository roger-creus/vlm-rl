# Backbone probe — Qwen/Qwen3.5-2B

## Inv-8 patch-coverage at multiple env resolutions

Both checks PASS: the processor emits a non-zero image-token count and
the token count scales with the pixel budget as expected.

| env / resolution         | min_pixels = max_pixels | image tokens | input_ids length |
|--------------------------|--------------------------|--------------|------------------|
| VizdoomBasic (320×240)   | 76800                    | 280          | 88               |
| ALE/Pong (160×210)       | 33600                    | 120          | 48               |

Regenerate with:

```bash
python -m scripts.probe_backbone --backbone Qwen/Qwen3.5-2B \
    --min-pixels 76800 --max-pixels 76800 --width 320 --height 240
python -m scripts.probe_backbone --backbone Qwen/Qwen3.5-2B \
    --min-pixels 33600 --max-pixels 33600 --width 160 --height 210
```
