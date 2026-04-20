# Checkpointing

Master-spec §10 describes the full save/resume system:

- Atomic write + rename + integrity manifest.
- Directory per checkpoint — model, optimizer, training, envs, logging, config, manifest.
- Retention: last 3 + every 10th + first + last-known-good.
- SIGTERM handler flushes inside 60 seconds.
- Resume gate runs Inv-7 + Inv-12 before continuing.

Implementation lives in `src/cleanrl_vlm/training/checkpoint.py` (populated by /loop).
