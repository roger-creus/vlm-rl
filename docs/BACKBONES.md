# Backbones

Actual backbones available on Hugging Face (as of 2026-04-20 — see
`docs/superpowers/specs/amendments/2026-04-20-backbone-names-correction.md`
for the deviation from master-spec §3's Qwen3.5-VL naming):

| name | params | thinking default | tier | notes |
|------|--------|------------------|------|-------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2 B | off | Tier-1 smoke + debug | smallest Qwen3-VL; replaces spec's non-existent 0.8B |
| `Qwen/Qwen3-VL-4B-Instruct` | 4 B | off | Tier-2 paper runs (hero) | spec's target 4B size |
| `Qwen/Qwen3-VL-4B-Thinking` | 4 B | **on** | Tier-2 thinking ablation | paper-ablations entry |
| `Qwen/Qwen3-VL-8B-Instruct` | 8 B | off | Tier-2 scale-up ablation | bigger-is-better check |

All load via `AutoModelForCausalLM` + `AutoProcessor`. Apache-2.0.

## Architecture (to be confirmed per S-5 web-fetch on first load)

Master-spec §3 described Qwen3.5-VL as hybrid
Gated-DeltaNet + Gated-Attention + sparse MoE. Qwen3-VL does **not**
publicly advertise the same hybrid; the /loop agent updates this table
the first time each backbone is probed under `docs/backbone_probes/`.

Onboarding a new backbone = master-spec §11 S-6 ritual: registry entry +
processor probe + §4 image-input probes + Inv-1..14 pass +
`docs/backbone_probes/<name>.md`.
