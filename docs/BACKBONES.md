# Backbones

| name | params | context | thinking-mode default | notes |
|------|--------|---------|-----------------------|-------|
| `Qwen/Qwen3.5-VL-0.8B` | 0.8B | 262 144 | off (loop-prone) | Tier-1 smoke + debug |
| `Qwen/Qwen3.5-VL-4B`   | 4B   | 262 144 | **on**           | Tier-2 paper runs    |

Hybrid architecture: Gated-DeltaNet (linear attention) + Gated-Attention + sparse MoE. Both load via `AutoModelForCausalLM` + `AutoProcessor`. Apache-2.0.

Onboarding a new backbone = master-spec §11 S-6 ritual: registry entry + processor probe + §4 image-input probes + Inv-1..14 pass + `docs/backbone_probes/<name>.md`.
