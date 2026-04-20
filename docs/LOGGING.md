# Logging

Three parallel sinks (master-spec §9):

- **Rich console** — live colored dashboard, auto-off in headless/CI.
- **W&B** — opt-in via `--track`; mirrors every scalar + histograms + eval-episode videos.
- **CSV** — always on; `runs/<name>/metrics.csv` one row per step with every scalar.

Plus `runs/<name>/histograms.parquet` for distributions and `runs/<name>/manifest.json` for the run config + git SHA + pip freeze.

Implementation lives in `src/cleanrl_vlm/training/logging.py` (populated by /loop).
