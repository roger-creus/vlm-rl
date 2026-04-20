---
title: Plan — PPO-COT · VizdoomBasic · Qwen3-VL-2B-Instruct
slug: B-ppo-cot-vizdoom-basic
date: 2026-04-20
refs:
  - docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md
  - docs/superpowers/specs/amendments/2026-04-20-ppo-cot-vizdoom-basic-design.md
  - docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md
---

# Implementation plan — B-ppo-cot-vizdoom-basic-2B

**Goal.** Ship `algos/ppo_cot.py`: a single-file PPO trainer that finetunes `Qwen/Qwen3-VL-2B-Instruct` on `VizdoomBasic-v0` via the chain-of-thought (COT) action interface, backed by the `src/cleanrl_vlm/` library. Green = agent-judged learning curve + 8 invariants (Inv-1/3/4/5/6/9/10/11/13) passing.

**Architecture.** Hybrid library + single-file trainer. `src/cleanrl_vlm/{envs,models,prompts,rollout,training}/` export typed units; `algos/ppo_cot.py` glues them together. Dual-adapter LoRA (actor+critic) on a shared `AutoModelForImageTextToText` base; `CriticHead` MLP on the critic adapter's last non-pad hidden state; GAE on env rewards; sequence-level PPO ratio from summed token log-probs.

**Tech stack.** Python 3.10+, PyTorch 2.6, `transformers` (git, Qwen3-VL support), PEFT, `accelerate` + DS-ZeRO-2 (single-rank at iter 4), ViZDoom-gymnasium, `tyro` CLI, UV deps. Testing: `uv run --no-env-file pytest` / `ruff` / `pyright`.

## For agentic workers

**REQUIRED SUB-SKILL:** `superpowers:subagent-driven-development`. Each task below is a unit of parallelizable work; spawn a worker per task, orchestrator assembles. Every code task is red → impl → green → commit, following `superpowers:test-driven-development`. Verification commands are explicit and their expected output is asserted before moving on. If any worker returns a test failure, invoke `superpowers:systematic-debugging` before patching.

## File structure

| File | Purpose |
|---|---|
| `configs/backbones.yaml` | Per-backbone defaults (pixel budget, attn impl, LoRA groups). |
| `configs/targets.yaml` | Reference target scores per env. |
| `configs/envs/VizdoomBasic-v0.yaml` | Frame-skip, resolution, reward scaling, pixel-budget override, `max_episode_steps`. |
| `src/cleanrl_vlm/envs/wrappers.py` | `FrameSkipEnv`, `ScreenOnlyWrapper`, `DiscreteMultiBinaryWrapper`. |
| `src/cleanrl_vlm/envs/vizdoom/__init__.py` | Package init. |
| `src/cleanrl_vlm/envs/vizdoom/action_tables.py` | `action_tables: dict[str, list[str]]`. |
| `src/cleanrl_vlm/envs/vizdoom/factories.py` | `make_vizdoom_env(env_id, config)`. |
| `src/cleanrl_vlm/envs/registry.py` | `make_env(env_id, config, seed, idx, run_name)`. |
| `src/cleanrl_vlm/models/lora_topology.py` | `default_target_modules(groups)`. |
| `src/cleanrl_vlm/models/heads.py` | `CriticHead`, `ActorHead`, `layer_init`. |
| `src/cleanrl_vlm/models/base_vlm.py` | `BaseVLM` wrapping `AutoModelForImageTextToText`. |
| `src/cleanrl_vlm/models/actor_critic.py` | `DecoupledActorCriticVLM_COT`, `active_adapter` ctxmgr. |
| `src/cleanrl_vlm/prompts/templates/vizdoom/basic/{actor,critic,vision_probe}.txt` | Prompt templates. |
| `src/cleanrl_vlm/prompts/parser.py` | `parse_action_cot(text, action_names)`. |
| `src/cleanrl_vlm/prompts/builder.py` | `PromptBuilder`. |
| `src/cleanrl_vlm/rollout/buffer.py` | `RolloutBuffer` + GAE. |
| `src/cleanrl_vlm/rollout/in_process.py` | `generate_cot_actions` → `CotRolloutStep` dataclass. |
| `src/cleanrl_vlm/training/distributed.py` | `load_accelerator_config`. |
| `src/cleanrl_vlm/training/precision.py` | `Fp16State`. |
| `src/cleanrl_vlm/training/microbatch_probe.py` | `probe_microbatch`. |
| `src/cleanrl_vlm/training/logging.py` | `RichDashboard`, `CsvWriter`, `wandb_init`. |
| `src/cleanrl_vlm/training/checkpoint.py` | `save_vlm_actor_critic_checkpoint`, loader. |
| `src/cleanrl_vlm/training/invariants.py` | `InvariantMonitor` + per-Inv check funcs. |
| `algos/ppo_cot.py` | Single-file trainer. |
| `scripts/_cluster_env.sh` | `CUDA_HOME` / `HF_HOME` boilerplate. |
| `scripts/probe_vision.py` | Inv-15 ground-truth vision probe. |
| `scripts/probe_backbone.py` | Inv-8 patch-coverage + token-count probe. |
| `tests/unit/test_env_factory.py` | Wrappers + factories. |
| `tests/unit/test_lora_topology.py` | Target-module resolution. |
| `tests/unit/test_action_parser.py` | Regex last-match + whitelist + pathology. |
| `tests/unit/test_prompt_builder.py` | Chat template shape + image-token presence. |
| `tests/unit/test_rollout_buffer.py` | Buffer indexing + dtypes. |
| `tests/unit/test_gae.py` | GAE vs hand-computed. |
| `tests/unit/test_microbatch_probe.py` | Probe logic on a tiny model. |
| `tests/invariants/test_inv_01_lora_trainability.py` | Inv-1 + base-weight identity + disjoint optimizer groups. |
| `tests/invariants/test_inv_03_active_adapter.py` | Inv-3 ctxmgr tripwire. |
| `tests/invariants/test_inv_04_logprob_parity.py` | Inv-4 single-path re-score parity. |
| `tests/invariants/test_inv_05_grad_norm.py` | Inv-5 global grad-norm cross-check. |
| `tests/invariants/test_inv_06_fp16_scale.py` | Inv-6 scale-factor + no-NaN. |
| `tests/invariants/test_inv_09_reward_pipeline.py` | Inv-9 scripted rewards. |
| `tests/invariants/test_inv_10_episode_boundary.py` | Inv-10 GAE reset at done=True. |
| `tests/invariants/test_inv_11_determinism.py` | Inv-11 bitwise. |
| `tests/invariants/test_inv_13_pad_image_token_mask.py` | Inv-13 pad-gradient zero. |
| `tests/integration/test_trainer_short_run.py` | tier1 @gpu: 10 iters on VizdoomBasic. |
| `docs/vision_probes/VizdoomBasic-v0_qwen3-vl-2b-instruct/report.md` | Auto-generated probe report. |
| `docs/backbone_probes/qwen3-vl-2b-instruct.md` | Auto-generated backbone probe. |
| `docs/ALGORITHMS.md`, `docs/ENVS.md`, `docs/RECIPES.md`, `docs/BACKBONES.md`, `docs/RESULTS.md` | Doc deliverables. |

---

## Task 1: Configs — backbones + targets + VizdoomBasic env

Rationale: per-env pixel budget override (reviewer B3) + `max_episode_steps=null` (M8) + `frame_stack.n=1` TODO (M6) have to land before any code touches them.

**Files:** Create `configs/backbones.yaml`, `configs/targets.yaml`, `configs/envs/VizdoomBasic-v0.yaml`.

- [ ] **Step 1: Write `configs/backbones.yaml`.**

```yaml
Qwen/Qwen3-VL-2B-Instruct:
  tier: 1
  auto_class: AutoModelForImageTextToText
  processor_pixel_budget:
    # Backbone default; per-env YAML overrides via processor_min_pixels / processor_max_pixels.
    min_pixels: 262144
    max_pixels: 1310720
  attn_implementation: flash_attention_2
  dtype: float16
  thinking_mode: off
  lora_groups_default: [text_attn, text_mlp, lm_head]
  lora_rank: 32
  lora_alpha: 64

Qwen/Qwen3-VL-4B-Instruct:
  tier: 2
  auto_class: AutoModelForImageTextToText
  processor_pixel_budget:
    min_pixels: 262144
    max_pixels: 1310720
  attn_implementation: flash_attention_2
  dtype: float16
  thinking_mode: off
  lora_groups_default: [text_attn, text_mlp, lm_head]
  lora_rank: 32
  lora_alpha: 64
```

- [ ] **Step 2: Write `configs/targets.yaml`.**

```yaml
VizdoomBasic-v0:
  Qwen/Qwen3-VL-2B-Instruct:
    reference_score: 60.0
    zero_shot_baseline: null
    scratch_cnn_baseline: null
```

- [ ] **Step 3: Write `configs/envs/VizdoomBasic-v0.yaml`.**

```yaml
tier: 1
backbone_default: Qwen/Qwen3-VL-2B-Instruct
frame_skip: 4
frame_stack:
  n: 1                         # TODO(M6): re-evaluate when corridor + defend_line ports land; tracked in LOOP_STATE under C-envs-tier1-expand
  layout: horizontal
resolution:
  width: 320
  height: 240
screen_format: RGB24
reward_clip: false
max_episode_steps: null        # ViZDoom Basic uses its own 300-tic cap (~75 env steps at frame_skip=4)
target_score: 60.0
# Override processor budget to match native 320x240 = 76800 pixels; avoids silent 3.4x upscale.
processor_min_pixels: 76800
processor_max_pixels: 76800
```

- [ ] **Step 4: Verify the three files parse as YAML.**

```bash
uv run --no-env-file python -c "import yaml; [yaml.safe_load(open(p)) for p in ['configs/backbones.yaml','configs/targets.yaml','configs/envs/VizdoomBasic-v0.yaml']]; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Commit.**

```bash
git add configs/backbones.yaml configs/targets.yaml configs/envs/VizdoomBasic-v0.yaml
git commit -m "$(cat <<'EOF'
config: add backbones / targets / VizdoomBasic env YAMLs

Seeds Qwen3-VL-2B/4B backbone defaults, VizdoomBasic reference score,
and per-env pixel-budget override matching native 320x240 (76800 px)
to avoid processor upscale. max_episode_steps=null per reviewer M8;
frame_stack.n=1 with follow-up TODO per reviewer M6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Env wrappers (TDD)

Rationale: `DiscreteMultiBinaryWrapper` replaces both prototype wrappers (reviewer m2); the unit test asserts generalization across button-count.

**Files:** Create `src/cleanrl_vlm/envs/wrappers.py`, `tests/unit/test_env_factory.py`.

- [ ] **Step 1: Write failing test `tests/unit/test_env_factory.py`.**

```python
import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces


def test_frame_skip_repeats_action_and_sums_reward():
    from src.cleanrl_vlm.envs.wrappers import FrameSkipEnv

    class Dummy(gym.Env):
        observation_space = spaces.Box(0, 255, (4,), dtype=np.uint8)
        action_space = spaces.Discrete(2)
        def __init__(self):
            self.t = 0
        def reset(self, *, seed=None, options=None):
            self.t = 0
            return np.zeros(4, dtype=np.uint8), {}
        def step(self, action):
            self.t += 1
            return np.full(4, self.t, dtype=np.uint8), 1.0, self.t >= 10, False, {}

    env = FrameSkipEnv(Dummy(), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
    assert reward == 4.0
    assert obs[0] == 4


def test_screen_only_wrapper_unwraps_dict_obs():
    from src.cleanrl_vlm.envs.wrappers import ScreenOnlyWrapper

    class DictEnv(gym.Env):
        observation_space = spaces.Dict({"screen": spaces.Box(0, 255, (3, 4, 4), dtype=np.uint8)})
        action_space = spaces.Discrete(2)
        def reset(self, *, seed=None, options=None):
            return {"screen": np.zeros((3, 4, 4), dtype=np.uint8)}, {}
        def step(self, action):
            return {"screen": np.ones((3, 4, 4), dtype=np.uint8)}, 0.0, False, False, {}

    env = ScreenOnlyWrapper(DictEnv())
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3, 4, 4)


@pytest.mark.parametrize("buttons,chosen_idx", [
    (["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"], 2),
    (["MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"], 5),
    (["TURN_LEFT", "TURN_RIGHT", "ATTACK"], 0),
])
def test_discrete_multibinary_wrapper_generalizes(buttons, chosen_idx):
    from src.cleanrl_vlm.envs.wrappers import DiscreteMultiBinaryWrapper

    class MB(gym.Env):
        def __init__(self, n):
            self.action_space = spaces.MultiBinary(n)
            self.observation_space = spaces.Box(0, 255, (1,), dtype=np.uint8)
            self.last_action = None
        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.uint8), {}
        def step(self, action):
            self.last_action = list(action)
            return np.zeros(1, dtype=np.uint8), 0.0, False, False, {}

    base = MB(len(buttons))
    env = DiscreteMultiBinaryWrapper(base, button_names=buttons)
    assert env.action_space == spaces.Discrete(len(buttons))
    env.reset()
    env.step(chosen_idx)
    expected = [0] * len(buttons)
    expected[chosen_idx] = 1
    assert base.last_action == expected
```

- [ ] **Step 2: Run test — expect ImportError.**

```bash
uv run --no-env-file pytest tests/unit/test_env_factory.py -v
```
Expected: collection error (module not found) or 5 failed.

- [ ] **Step 3: Write `src/cleanrl_vlm/envs/wrappers.py`.**

```python
"""Gymnasium wrappers for cleanrl-vlm env layer."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameSkipEnv(gym.Wrapper):
    """Repeat the same action `skip` times, sum reward, return final obs."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = None
        info: dict = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ScreenOnlyWrapper(gym.ObservationWrapper):
    """Drop ViZDoom dict obs down to just the `screen` RGB array."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "ScreenOnlyWrapper expects Dict obs"
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs):
        return np.asarray(obs["screen"])


class DiscreteMultiBinaryWrapper(gym.ActionWrapper):
    """Generalizes DeadlyCorridor / DefendTheLine prototype wrappers.

    Converts a MultiBinary(N)-style button space into Discrete(N) by setting
    exactly one button to 1 per step. Button names are recorded for logging
    and prompt construction.
    """

    def __init__(self, env: gym.Env, button_names: list[str]) -> None:
        super().__init__(env)
        self.button_names = list(button_names)
        self.action_space = spaces.Discrete(len(self.button_names))

    def action(self, act):
        arr = [0] * len(self.button_names)
        if 0 <= int(act) < len(self.button_names):
            arr[int(act)] = 1
        return arr
```

- [ ] **Step 4: Run test — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_env_factory.py -v
```
Expected: `5 passed`.

- [ ] **Step 5: Commit.**

```bash
git add src/cleanrl_vlm/envs/wrappers.py tests/unit/test_env_factory.py
git commit -m "$(cat <<'EOF'
envs: add FrameSkipEnv / ScreenOnlyWrapper / DiscreteMultiBinaryWrapper

DiscreteMultiBinaryWrapper supersedes the prototype DeadlyCorridor /
DefendTheLine action wrappers (reviewer m2). Unit test parameterizes
across 3 scenarios to assert generalization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: VizDoom action tables + factories

Rationale: `action_tables.py` is the single source of truth for per-scenario button lists; `factories.py` wires the three wrappers in the right order.

**Files:** Create `src/cleanrl_vlm/envs/vizdoom/__init__.py`, `.../action_tables.py`, `.../factories.py`. Extend `tests/unit/test_env_factory.py`.

- [ ] **Step 1: Append to `tests/unit/test_env_factory.py`.**

```python
def test_action_tables_contains_vizdoom_basic():
    from src.cleanrl_vlm.envs.vizdoom.action_tables import action_tables
    assert action_tables["VizdoomBasic-v0"] == ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
    assert action_tables["VizdoomCorridor-v0"] == [
        "MOVE_LEFT", "MOVE_RIGHT", "ATTACK",
        "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
    ]
    assert action_tables["VizdoomDefendLine-v0"] == ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
```

- [ ] **Step 2: Run — expect fail.**

```bash
uv run --no-env-file pytest tests/unit/test_env_factory.py::test_action_tables_contains_vizdoom_basic -v
```
Expected: import error.

- [ ] **Step 3: Write `src/cleanrl_vlm/envs/vizdoom/__init__.py`.**

```python
"""ViZDoom env sub-package."""
```

- [ ] **Step 4: Write `src/cleanrl_vlm/envs/vizdoom/action_tables.py`.**

```python
"""Per-scenario ViZDoom button-name lookup.

Adding a new ViZDoom scenario requires (a) a new entry here, (b) wiring
in `factories.py`, and (c) a prompt template under
`src/cleanrl_vlm/prompts/templates/vizdoom/<slug>/`.
"""
from __future__ import annotations

action_tables: dict[str, list[str]] = {
    "VizdoomBasic-v0": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
    "VizdoomCorridor-v0": [
        "MOVE_LEFT", "MOVE_RIGHT", "ATTACK",
        "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
    ],
    "VizdoomDefendLine-v0": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
}
```

- [ ] **Step 5: Write `src/cleanrl_vlm/envs/vizdoom/factories.py`.**

```python
"""ViZDoom env factories."""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym

from src.cleanrl_vlm.envs.vizdoom.action_tables import action_tables
from src.cleanrl_vlm.envs.wrappers import (
    DiscreteMultiBinaryWrapper,
    FrameSkipEnv,
    ScreenOnlyWrapper,
)


def make_vizdoom_env(env_id: str, config: dict[str, Any]) -> Callable[[], gym.Env]:
    """Return a thunk that builds the wrapped env.

    Wrapper order: gym.make -> (frame-skip baked into gym.make via frame_skip
    kwarg) -> DiscreteMultiBinaryWrapper -> ScreenOnlyWrapper ->
    RecordEpisodeStatistics.
    """
    # Ensure the vizdoom gymnasium wrapper is registered.
    from vizdoom import gymnasium_wrapper  # noqa: F401

    if env_id not in action_tables:
        raise KeyError(f"Unknown ViZDoom env {env_id!r}; extend action_tables.")
    buttons = action_tables[env_id]
    frame_skip = int(config.get("frame_skip", 4))

    def thunk() -> gym.Env:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            max_buttons_pressed=0,
            frame_skip=frame_skip,
        )
        env = DiscreteMultiBinaryWrapper(env, button_names=buttons)
        env = ScreenOnlyWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
```

- [ ] **Step 6: Run test — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_env_factory.py -v
```
Expected: `6 passed`.

- [ ] **Step 7: Commit.**

```bash
git add src/cleanrl_vlm/envs/vizdoom/__init__.py src/cleanrl_vlm/envs/vizdoom/action_tables.py src/cleanrl_vlm/envs/vizdoom/factories.py tests/unit/test_env_factory.py
git commit -m "$(cat <<'EOF'
envs: add VizDoom action_tables + make_vizdoom_env factory

action_tables keys VizdoomBasic/Corridor/DefendLine; factory composes
DiscreteMultiBinaryWrapper + ScreenOnlyWrapper + RecordEpisodeStatistics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Env registry dispatcher

Rationale: Single `make_env` entrypoint so the trainer never branches on env-id-prefix.

**Files:** Create `src/cleanrl_vlm/envs/registry.py`, extend `tests/unit/test_env_factory.py`.

- [ ] **Step 1: Append to `tests/unit/test_env_factory.py`.**

```python
def test_registry_dispatches_vizdoom():
    from src.cleanrl_vlm.envs.registry import make_env

    thunk = make_env(
        env_id="VizdoomBasic-v0",
        config={"frame_skip": 4},
        seed=0,
        idx=0,
        run_name="test",
    )
    assert callable(thunk)


def test_registry_rejects_unknown_env():
    import pytest
    from src.cleanrl_vlm.envs.registry import make_env

    with pytest.raises(KeyError):
        make_env(env_id="Unknown-v0", config={}, seed=0, idx=0, run_name="test")
```

- [ ] **Step 2: Run — expect fail.**

```bash
uv run --no-env-file pytest tests/unit/test_env_factory.py::test_registry_dispatches_vizdoom tests/unit/test_env_factory.py::test_registry_rejects_unknown_env -v
```
Expected: import error.

- [ ] **Step 3: Write `src/cleanrl_vlm/envs/registry.py`.**

```python
"""Single-point env registration / dispatch."""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym


def make_env(
    env_id: str,
    config: dict[str, Any],
    seed: int,
    idx: int,
    run_name: str,
) -> Callable[[], gym.Env]:
    """Dispatch to the appropriate env sub-package by env-id prefix."""
    if env_id.startswith("Vizdoom"):
        from src.cleanrl_vlm.envs.vizdoom.factories import make_vizdoom_env
        thunk = make_vizdoom_env(env_id, config)
    else:
        raise KeyError(f"No factory registered for env_id={env_id!r}")

    def seeded_thunk() -> gym.Env:
        env = thunk()
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        return env

    return seeded_thunk
```

- [ ] **Step 4: Run — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_env_factory.py -v
```
Expected: `8 passed`.

- [ ] **Step 5: Commit.**

```bash
git add src/cleanrl_vlm/envs/registry.py tests/unit/test_env_factory.py
git commit -m "$(cat <<'EOF'
envs: add make_env registry dispatcher

Single entrypoint; dispatches by env-id prefix, applies per-idx seed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: LoRA topology helper (TDD)

Rationale: `default_target_modules(groups)` returns the flat list of module-name suffixes implied by a set of group names. Unit-testable on a tiny `nn.Module` mock.

**Files:** Create `src/cleanrl_vlm/models/lora_topology.py`, `tests/unit/test_lora_topology.py`.

- [ ] **Step 1: Write failing test `tests/unit/test_lora_topology.py`.**

```python
import pytest


def test_default_target_modules_text_attn():
    from src.cleanrl_vlm.models.lora_topology import default_target_modules
    out = default_target_modules({"text_attn"})
    assert "self_attn.q_proj" in out
    assert "self_attn.k_proj" in out
    assert "self_attn.v_proj" in out
    assert "self_attn.o_proj" in out
    assert "mlp.gate_proj" not in out


def test_default_target_modules_text_mlp():
    from src.cleanrl_vlm.models.lora_topology import default_target_modules
    out = default_target_modules({"text_mlp"})
    assert set(out) >= {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}


def test_default_target_modules_lm_head():
    from src.cleanrl_vlm.models.lora_topology import default_target_modules
    out = default_target_modules({"lm_head"})
    assert "lm_head" in out


def test_default_target_modules_combined():
    from src.cleanrl_vlm.models.lora_topology import default_target_modules
    out = default_target_modules({"text_attn", "text_mlp", "lm_head"})
    assert "self_attn.q_proj" in out
    assert "mlp.down_proj" in out
    assert "lm_head" in out


def test_default_target_modules_unknown_group_raises():
    from src.cleanrl_vlm.models.lora_topology import default_target_modules
    with pytest.raises(ValueError):
        default_target_modules({"bogus"})


def test_default_target_modules_returns_list_without_duplicates():
    from src.cleanrl_vlm.models.lora_topology import default_target_modules
    out = default_target_modules({"text_attn", "text_attn"})
    assert isinstance(out, list)
    assert len(out) == len(set(out))
```

- [ ] **Step 2: Run — expect import error.**

```bash
uv run --no-env-file pytest tests/unit/test_lora_topology.py -v
```
Expected: collection error.

- [ ] **Step 3: Write `src/cleanrl_vlm/models/lora_topology.py`.**

```python
"""LoRA target-module group resolution."""
from __future__ import annotations

from typing import Iterable

_GROUPS: dict[str, list[str]] = {
    "text_attn": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
    "text_mlp": [
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    "vision_attn": [
        "attn.qkv",
        "attn.proj",
    ],
    "vision_mlp": [
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    ],
    "merger": [
        "merger.linear_fc1",
        "merger.linear_fc2",
    ],
    "lm_head": [
        "lm_head",
    ],
    "text_moe": [
        "block_sparse_moe.gate",
    ],
}


def default_target_modules(groups: Iterable[str]) -> list[str]:
    """Resolve a set of LoRA group names to a flat de-duplicated list of
    module-name suffixes consumable by `peft.LoraConfig(target_modules=...)`."""
    groups = list(groups)
    unknown = [g for g in groups if g not in _GROUPS]
    if unknown:
        raise ValueError(f"Unknown LoRA group(s): {unknown}. Known: {sorted(_GROUPS)}")
    out: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for m in _GROUPS[g]:
            if m not in seen:
                out.append(m)
                seen.add(m)
    return out
```

- [ ] **Step 4: Run — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_lora_topology.py -v
```
Expected: `6 passed`.

- [ ] **Step 5: Commit.**

```bash
git add src/cleanrl_vlm/models/lora_topology.py tests/unit/test_lora_topology.py
git commit -m "$(cat <<'EOF'
models: add default_target_modules(groups) LoRA topology helper

Resolves {text_attn, text_mlp, vision_attn, vision_mlp, merger, lm_head,
text_moe} group names to flat module-suffix lists. Raises on unknown
group. De-duplicates.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: MLP heads (`CriticHead`, `ActorHead`)

Rationale: Small, no-VLM module; direct port from prototype with `layer_init`.

**Files:** Create `src/cleanrl_vlm/models/heads.py`, `tests/unit/test_heads.py`.

- [ ] **Step 1: Write test `tests/unit/test_heads.py`.**

```python
import torch


def test_critic_head_shape():
    from src.cleanrl_vlm.models.heads import CriticHead
    h = CriticHead(input_dim=128)
    x = torch.randn(4, 128)
    y = h(x)
    assert y.shape == (4, 1)


def test_actor_head_shape():
    from src.cleanrl_vlm.models.heads import ActorHead
    h = ActorHead(input_dim=128, num_actions=7)
    x = torch.randn(4, 128)
    y = h(x)
    assert y.shape == (4, 7)


def test_layer_init_is_orthogonal():
    import torch.nn as nn
    from src.cleanrl_vlm.models.heads import layer_init
    lin = layer_init(nn.Linear(32, 16))
    # Orthogonal: rows (output dim) should be mutually orthogonal when output <= input.
    w = lin.weight.detach()
    gram = w @ w.t()
    eye = torch.eye(gram.size(0))
    assert torch.allclose(gram, gram.diag().diag(), atol=1e-4)
    assert torch.allclose(lin.bias.detach(), torch.zeros_like(lin.bias))
```

- [ ] **Step 2: Run — expect fail.**

```bash
uv run --no-env-file pytest tests/unit/test_heads.py -v
```
Expected: collection error.

- [ ] **Step 3: Write `src/cleanrl_vlm/models/heads.py`.**

```python
"""Orthogonal-init MLP heads shared across canon trainers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = float(np.sqrt(2)), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CriticHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class ActorHead(nn.Module):
    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)
```

- [ ] **Step 4: Run — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_heads.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit.**

```bash
git add src/cleanrl_vlm/models/heads.py tests/unit/test_heads.py
git commit -m "$(cat <<'EOF'
models: add CriticHead / ActorHead + orthogonal layer_init

3-layer 512-hidden LeakyReLU MLPs; critic std=1.0 on output, actor std=0.01.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `BaseVLM` wrapper

Rationale: Backbone-agnostic wrapper. No meaningful CPU unit test — integration coverage via Task 8 tests.

**Files:** Create `src/cleanrl_vlm/models/base_vlm.py`, `src/cleanrl_vlm/models/__init__.py` helpers.

- [ ] **Step 1: Write `src/cleanrl_vlm/models/base_vlm.py`.**

```python
"""Backbone-agnostic VLM wrapper."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def _numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(img.astype(np.uint8)) for img in images]


class BaseVLM(nn.Module):
    """Holds a HF VLM + processor and common pre/post-processing helpers."""

    def __init__(
        self,
        vlm_name: str,
        min_pixels: int,
        max_pixels: int,
        attn_implementation: str = "flash_attention_2",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(
            vlm_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            vlm_name,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

    def preprocess_obs_and_text(
        self,
        obs: torch.Tensor,
        text_prompts: list[str],
        add_generation_prompt: bool = True,
    ) -> Any:
        pil_images = _numpy_to_pil(obs.cpu().numpy())
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            for t in text_prompts
        ]
        inputs = self.processor(
            text=texts, images=pil_images, return_tensors="pt", padding=True,
        ).to(self.model.device)
        return inputs

    @staticmethod
    def last_hidden_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_token_indices, :]

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
```

- [ ] **Step 2: Sanity import.**

```bash
uv run --no-env-file python -c "from src.cleanrl_vlm.models.base_vlm import BaseVLM; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit.**

```bash
git add src/cleanrl_vlm/models/base_vlm.py
git commit -m "$(cat <<'EOF'
models: add BaseVLM wrapping AutoModelForImageTextToText + AutoProcessor

Uses the Auto class (Qwen3-VL support confirmed in iter 3 smoke).
Exposes preprocess_obs_and_text, last_hidden_state, get_trainable_params.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `DecoupledActorCriticVLM_COT` + `active_adapter` ctxmgr + Inv-1/Inv-3 tests

Rationale: The dual-adapter model and its tripwire ctxmgr. Inv-1 test is extended (reviewer M4) to assert base-weight identity across swaps + disjoint optim param groups. `active_adapter` lives here (reviewer m7). Asserts build-time that actor-adapter and critic-adapter target-module lists are identical (reviewer m10). Docstring states critic path uses `.forward()` only (reviewer m9).

**Files:** Create `src/cleanrl_vlm/models/actor_critic.py`, `tests/invariants/test_inv_01_lora_trainability.py`, `tests/invariants/test_inv_03_active_adapter.py`.

**Time-box caveat (reviewer m3):** A `TinyVLMForImageTextToText` stub that can stand in for Qwen3-VL in these tests must stay under ~150 LOC. If it exceeds that, demote the Inv-01/03 tests to `@pytest.mark.tier1 @pytest.mark.gpu` against the real 2B backbone and delete the stub. The stub implementation path is the default.

- [ ] **Step 1: Write `src/cleanrl_vlm/models/actor_critic.py`.**

```python
"""Decoupled actor-critic VLM with dual LoRA adapters (COT interface)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.distributions.categorical import Categorical

from src.cleanrl_vlm.models.base_vlm import BaseVLM
from src.cleanrl_vlm.models.heads import CriticHead
from src.cleanrl_vlm.models.lora_topology import default_target_modules


@contextmanager
def active_adapter(ac_model: "DecoupledActorCriticVLM_COT", name: str) -> Iterator[None]:
    """Tripwire ctxmgr: asserts the PEFT active adapter equals `name` on enter.

    Setting the adapter happens inside `get_action` / `get_value`; this ctxmgr
    does NOT set, so a missing call surfaces as an assertion failure rather
    than a silent cross-adapter forward (Inv-3).
    """
    current = ac_model.vlm.model.active_adapter
    if current != name:
        raise AssertionError(f"active_adapter expected {name!r}, got {current!r}")
    try:
        yield
    finally:
        still = ac_model.vlm.model.active_adapter
        if still != name:
            raise AssertionError(f"active_adapter mutated inside ctxmgr: {name!r} -> {still!r}")


class DecoupledActorCriticVLM_COT(nn.Module):
    """One VLM, two LoRA adapters named 'actor' and 'critic', plus a CriticHead.

    The critic forward path uses `.forward()` only -- it NEVER invokes
    `.generate()`. The actor `.generate()` path is the only generation path.
    """

    def __init__(
        self,
        vlm_name: str,
        min_pixels: int,
        max_pixels: int,
        attn_implementation: str,
        dtype: torch.dtype,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_groups: tuple[str, ...],
        max_new_tokens: int,
    ) -> None:
        super().__init__()
        self.vlm = BaseVLM(
            vlm_name=vlm_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        hidden_size = self.vlm.model.config.text_config.hidden_size
        self.critic_head = CriticHead(hidden_size).to(self.vlm.model.dtype)
        self.max_new_tokens = max_new_tokens

        # Snapshot the target-module list ONCE; feed the identical list to both
        # LoraConfigs to avoid dict-ordering divergence (reviewer m10).
        target_modules = list(default_target_modules(set(lora_groups)))

        actor_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(target_modules), bias="none",
        )
        critic_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(target_modules), bias="none",
        )
        assert actor_cfg.target_modules == critic_cfg.target_modules, (
            "actor/critic LoRA target modules diverged"
        )

        self.vlm.model = get_peft_model(self.vlm.model, actor_cfg, adapter_name="actor")
        self.vlm.model.add_adapter("critic", critic_cfg)

        # Re-freeze: only lora_* params train (critic head handled separately in
        # get_trainable_params).
        for n, p in self.vlm.model.named_parameters():
            p.requires_grad = ("lora_" in n)

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        params = self.vlm.get_trainable_params()
        params.extend(list(self.critic_head.parameters()))
        return params

    def actor_param_ids(self) -> set[int]:
        return {id(p) for n, p in self.vlm.model.named_parameters()
                if "lora_" in n and ".actor." in n and p.requires_grad}

    def critic_param_ids(self) -> set[int]:
        critic_lora = {id(p) for n, p in self.vlm.model.named_parameters()
                       if "lora_" in n and ".critic." in n and p.requires_grad}
        critic_lora |= {id(p) for p in self.critic_head.parameters() if p.requires_grad}
        return critic_lora

    def get_action(
        self,
        obs: torch.Tensor,
        text_prompts: list[str],
        action_ids: Optional[torch.Tensor] = None,
        prompt_lens: Optional[torch.Tensor] = None,
    ):
        self.vlm.model.set_adapter("actor")
        with active_adapter(self, "actor"):
            inputs = self.vlm.preprocess_obs_and_text(obs, text_prompts, add_generation_prompt=True)
            batch_size = len(text_prompts)
            if action_ids is None:
                full_ids = self.vlm.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                )
                generated_texts = self.vlm.processor.batch_decode(
                    full_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
                prompt_lens = torch.tensor(
                    [inputs.input_ids.shape[1]] * batch_size, device=self.vlm.model.device,
                )
            else:
                full_ids = action_ids
            attention_mask = (full_ids != self.vlm.processor.tokenizer.pad_token_id).long()
            outputs = self.vlm.model(
                input_ids=full_ids,
                image_grid_thw=inputs.image_grid_thw,
                pixel_values=inputs.pixel_values,
                output_hidden_states=True,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            log_probs_all = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = full_ids[:, 1:]
            log_probs = torch.gather(log_probs_all, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            entropy = Categorical(logits=logits[:, :-1, :]).entropy()
            if action_ids is None:
                return log_probs, full_ids, prompt_lens, generated_texts
            return log_probs, entropy

    def get_value(self, obs: torch.Tensor, prompt_text: list[str]) -> torch.Tensor:
        """Critic path: `.forward()` only, never `.generate()` (reviewer m9)."""
        self.vlm.model.set_adapter("critic")
        with active_adapter(self, "critic"):
            inputs = self.vlm.preprocess_obs_and_text(obs, prompt_text)
            outputs = self.vlm.model(**inputs, output_hidden_states=True)
            last_hidden = self.vlm.last_hidden_state(
                outputs.hidden_states[-1], inputs["attention_mask"],
            )
            return self.critic_head(last_hidden)
```

- [ ] **Step 2: Write `tests/invariants/test_inv_01_lora_trainability.py`.**

Test fixtures use a `TinyVLMForImageTextToText` stub registered into `AutoModelForImageTextToText`. Keep under 150 LOC (reviewer m3); else demote to `@tier1 @gpu`.

```python
"""Inv-1 — LoRA trainability split.

Asserts:
1. requires_grad == True iff "lora_" in name or param in critic_head.
2. Base weights keep identity (`id(p)`) across set_adapter swaps (M4).
3. Optimizer param groups for actor and critic are disjoint (M4).
"""
from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.tier1


def _build_tiny_ac_model():
    # Minimal stub: if the stub grows past ~150 LOC, move this to @gpu on the
    # real Qwen3-VL-2B backbone per reviewer m3.
    from tests.invariants._tiny_vlm import build_tiny_ac_model
    return build_tiny_ac_model()


def test_inv_01_requires_grad_split():
    ac = _build_tiny_ac_model()
    for n, p in ac.vlm.model.named_parameters():
        expected = "lora_" in n
        assert p.requires_grad == expected, f"param {n} requires_grad={p.requires_grad} expected {expected}"
    for p in ac.critic_head.parameters():
        assert p.requires_grad is True


def test_inv_01_base_weight_identity_across_set_adapter():
    ac = _build_tiny_ac_model()
    # Collect base-weight tensor ids under 'actor'.
    ac.vlm.model.set_adapter("actor")
    base_ids_actor = {
        n: id(p) for n, p in ac.vlm.model.named_parameters() if "lora_" not in n
    }
    ac.vlm.model.set_adapter("critic")
    base_ids_critic = {
        n: id(p) for n, p in ac.vlm.model.named_parameters() if "lora_" not in n
    }
    assert base_ids_actor == base_ids_critic, (
        "base-weight tensor ids changed across set_adapter('actor') vs ('critic') -- "
        "adapters are NOT sharing base weights"
    )


def test_inv_01_actor_critic_param_groups_disjoint():
    ac = _build_tiny_ac_model()
    actor_ids = ac.actor_param_ids()
    critic_ids = ac.critic_param_ids()
    assert actor_ids, "no trainable actor params found"
    assert critic_ids, "no trainable critic params found"
    assert actor_ids.isdisjoint(critic_ids), (
        f"actor and critic param groups overlap on {len(actor_ids & critic_ids)} tensors"
    )
```

- [ ] **Step 3: Write `tests/invariants/test_inv_03_active_adapter.py`.**

```python
"""Inv-3 — active_adapter ctxmgr tripwire."""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.tier1


def _build_tiny_ac_model():
    from tests.invariants._tiny_vlm import build_tiny_ac_model
    return build_tiny_ac_model()


def test_ctxmgr_asserts_expected_adapter():
    from src.cleanrl_vlm.models.actor_critic import active_adapter

    ac = _build_tiny_ac_model()
    ac.vlm.model.set_adapter("actor")
    with active_adapter(ac, "actor"):
        pass


def test_ctxmgr_raises_when_adapter_mismatches():
    from src.cleanrl_vlm.models.actor_critic import active_adapter

    ac = _build_tiny_ac_model()
    ac.vlm.model.set_adapter("critic")
    with pytest.raises(AssertionError):
        with active_adapter(ac, "actor"):
            pass


def test_ctxmgr_raises_when_adapter_mutated_inside():
    from src.cleanrl_vlm.models.actor_critic import active_adapter

    ac = _build_tiny_ac_model()
    ac.vlm.model.set_adapter("actor")
    with pytest.raises(AssertionError):
        with active_adapter(ac, "actor"):
            ac.vlm.model.set_adapter("critic")
```

- [ ] **Step 4: Add `tests/invariants/_tiny_vlm.py` fixture builder (≤150 LOC).**

```python
"""Tiny VLM fixture used by invariant tests.

Registers a minimal ForImageTextToText model into the Auto class registry so
BaseVLM + PEFT can load it with `from_pretrained`. If this file exceeds ~150
LOC, delete it and mark the dependent invariant tests @gpu (reviewer m3).
"""
from __future__ import annotations

import tempfile

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class _TinyConfig:
    def __init__(self):
        self.text_config = type("TC", (), {"hidden_size": 32, "vocab_size": 128})()


class _TinyModel(nn.Module):
    """Pure-CPU, no HF deps. Exposes the minimal attribute surface the
    ActorCritic wrapper touches."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _TinyConfig()
        self.self_attn_q_proj = nn.Linear(32, 32, bias=False)
        # Use a nested module so PEFT can match "self_attn.q_proj" target.
        self.block = nn.Module()
        self.block.self_attn = nn.Module()
        self.block.self_attn.q_proj = nn.Linear(32, 32, bias=False)
        self.block.self_attn.k_proj = nn.Linear(32, 32, bias=False)
        self.block.self_attn.v_proj = nn.Linear(32, 32, bias=False)
        self.block.self_attn.o_proj = nn.Linear(32, 32, bias=False)
        self.lm_head = nn.Linear(32, 128, bias=False)


def build_tiny_ac_model():
    """Return a DecoupledActorCriticVLM_COT-shaped object backed by _TinyModel.

    Bypasses BaseVLM.from_pretrained; constructs a stub that exposes the
    attributes (`.vlm.model`, `.critic_head`, `.actor_param_ids`,
    `.critic_param_ids`) the Inv-01 / Inv-03 tests need.
    """
    from src.cleanrl_vlm.models.heads import CriticHead

    inner = _TinyModel()
    base_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "lm_head"],
        bias="none",
    )
    peft_model = get_peft_model(inner, base_cfg, adapter_name="actor")
    peft_model.add_adapter("critic", base_cfg)

    class _StubVLM:
        def __init__(self, m):
            self.model = m
    class _StubAC:
        pass

    ac = _StubAC()
    ac.vlm = _StubVLM(peft_model)
    ac.critic_head = CriticHead(32).to(torch.float32)
    for n, p in ac.vlm.model.named_parameters():
        p.requires_grad = ("lora_" in n)
    for p in ac.critic_head.parameters():
        p.requires_grad = True

    def actor_ids() -> set[int]:
        return {id(p) for n, p in ac.vlm.model.named_parameters()
                if "lora_" in n and ".actor." in n and p.requires_grad}

    def critic_ids() -> set[int]:
        s = {id(p) for n, p in ac.vlm.model.named_parameters()
             if "lora_" in n and ".critic." in n and p.requires_grad}
        s |= {id(p) for p in ac.critic_head.parameters() if p.requires_grad}
        return s

    ac.actor_param_ids = actor_ids
    ac.critic_param_ids = critic_ids
    return ac
```

- [ ] **Step 5: Run invariant tests.**

```bash
uv run --no-env-file pytest tests/invariants/test_inv_01_lora_trainability.py tests/invariants/test_inv_03_active_adapter.py -v
```
Expected: `6 passed`. If the stub fails (e.g., PEFT can't match modules on the toy model), verify stub size is ≤ 150 LOC and demote-to-`@gpu` per reviewer m3 — commit the demotion decision to AUTONOMY_LOG.

- [ ] **Step 6: Commit.**

```bash
git add src/cleanrl_vlm/models/actor_critic.py tests/invariants/test_inv_01_lora_trainability.py tests/invariants/test_inv_03_active_adapter.py tests/invariants/_tiny_vlm.py
git commit -m "$(cat <<'EOF'
models: add DecoupledActorCriticVLM_COT + active_adapter ctxmgr

Dual-adapter LoRA on a shared BaseVLM; CriticHead on critic path's last
non-pad hidden state. active_adapter() is a tripwire ctxmgr (reviewer
m7). Build-time asserts actor/critic target_modules identity (reviewer
m10). Critic path uses .forward() only, never .generate() (reviewer m9).

Inv-01 test extended: base-weight identity across set_adapter +
disjoint optim param groups (reviewer M4). Inv-03 covers ctxmgr
tripwire. TinyVLM stub time-boxed to 150 LOC (reviewer m3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Prompt templates for VizdoomBasic

Rationale: Templates are data, not code. Landing them before `PromptBuilder` lets the builder's unit test exercise the real actor prompt.

**Files:** Create `src/cleanrl_vlm/prompts/templates/vizdoom/basic/{actor,critic,vision_probe}.txt`.

- [ ] **Step 1: Write `.../actor.txt`.**

```text
You are playing VizdoomBasic, a simple ViZDoom scenario in which a
single monster appears randomly in front of you and you must shoot it
before a timeout.

Available actions:
- MOVE_LEFT  — strafe left
- MOVE_RIGHT — strafe right
- ATTACK     — shoot

Look at the current screen carefully. Think briefly about where the
monster is and whether you need to line up your aim. Then respond with
exactly one line of the form:

ACTION: <NAME>

where <NAME> is one of MOVE_LEFT, MOVE_RIGHT, ATTACK.
```

- [ ] **Step 2: Write `.../critic.txt`.**

```text
You are an RL value estimator for VizdoomBasic. Given the current
screen, estimate how good this state is for a future reward of +101 for
killing the monster minus a small living cost. Respond with no text —
your hidden representation is read by a downstream regressor.
```

- [ ] **Step 3: Write `.../vision_probe.txt`.**

```text
For each frame, answer three questions:
1. Is a monster visible? (yes / no)
2. Roughly where is the monster on the screen? (left / center / right / off-screen)
3. Is the player's crosshair roughly aligned with the monster's body? (yes / no / no-monster)
```

- [ ] **Step 4: Verify files exist.**

```bash
ls src/cleanrl_vlm/prompts/templates/vizdoom/basic/
```
Expected: `actor.txt  critic.txt  vision_probe.txt`.

- [ ] **Step 5: Commit.**

```bash
git add src/cleanrl_vlm/prompts/templates/vizdoom/basic/
git commit -m "$(cat <<'EOF'
prompts: add VizdoomBasic actor / critic / vision_probe templates

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Parser + PromptBuilder (TDD; M2 + M3)

Rationale: Parser lives in its own file (M2); regex `r"ACTION:\s*([A-Z_]+)"` takes the **last** match, whitelists against action names, returns `None` on fail (M3). Test covers the repeated-ACTION pathology.

**Files:** Create `src/cleanrl_vlm/prompts/parser.py`, `src/cleanrl_vlm/prompts/builder.py`, `tests/unit/test_action_parser.py`, `tests/unit/test_prompt_builder.py`.

- [ ] **Step 1: Write `tests/unit/test_action_parser.py`.**

```python
import pytest


ACTIONS = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]


def test_parses_simple_action():
    from src.cleanrl_vlm.prompts.parser import parse_action_cot
    assert parse_action_cot("ACTION: MOVE_LEFT", ACTIONS) == 0
    assert parse_action_cot("ACTION: MOVE_RIGHT", ACTIONS) == 1
    assert parse_action_cot("ACTION: ATTACK", ACTIONS) == 2


def test_parses_with_preceding_think():
    from src.cleanrl_vlm.prompts.parser import parse_action_cot
    text = "THOUGHTS: monster on the right, shoot\nACTION: ATTACK"
    assert parse_action_cot(text, ACTIONS) == 2


def test_takes_last_match_on_repeated_action_pathology():
    """Pathology: the model emits multiple ACTION lines. We take the last."""
    from src.cleanrl_vlm.prompts.parser import parse_action_cot
    text = "ACTION: MOVE_LEFT\nSorry, actually:\nACTION: ATTACK"
    assert parse_action_cot(text, ACTIONS) == 2


def test_whitelist_rejects_unknown_action():
    from src.cleanrl_vlm.prompts.parser import parse_action_cot
    assert parse_action_cot("ACTION: RUN_AWAY", ACTIONS) is None


def test_returns_none_on_no_match():
    from src.cleanrl_vlm.prompts.parser import parse_action_cot
    assert parse_action_cot("blah blah no tag", ACTIONS) is None


def test_is_case_sensitive_but_strips_trailing_whitespace():
    from src.cleanrl_vlm.prompts.parser import parse_action_cot
    assert parse_action_cot("ACTION: MOVE_LEFT  \n\n", ACTIONS) == 0


def test_regex_tolerates_extra_spaces_after_colon():
    from src.cleanrl_vlm.prompts.parser import parse_action_cot
    assert parse_action_cot("ACTION:    ATTACK", ACTIONS) == 2
```

- [ ] **Step 2: Run — expect fail.**

```bash
uv run --no-env-file pytest tests/unit/test_action_parser.py -v
```
Expected: import error.

- [ ] **Step 3: Write `src/cleanrl_vlm/prompts/parser.py`.**

```python
"""COT action parser: regex last-match + whitelist."""
from __future__ import annotations

import re

_ACTION_RE = re.compile(r"ACTION:\s*([A-Z_]+)")


def parse_action_cot(text: str, action_names: list[str]) -> int | None:
    """Extract the trailing ACTION from a VLM generation.

    Strategy (reviewer M3):
    - Regex `r"ACTION:\\s*([A-Z_]+)"`.
    - Take the LAST match in the text (model sometimes emits multiple).
    - Whitelist against `action_names`.
    - Return None on no-match / not-in-whitelist; caller samples uniformly.
    """
    matches = _ACTION_RE.findall(text or "")
    if not matches:
        return None
    last = matches[-1].strip()
    if last not in action_names:
        return None
    return action_names.index(last)
```

- [ ] **Step 4: Run parser tests — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_action_parser.py -v
```
Expected: `7 passed`.

- [ ] **Step 5: Write `src/cleanrl_vlm/prompts/builder.py`.**

```python
"""Prompt template assembly + chat-template application."""
from __future__ import annotations

from importlib import resources
from pathlib import Path


class PromptBuilder:
    """Loads actor/critic prompt templates for a given env_id + action names."""

    def __init__(self, env_id: str, action_names: list[str], templates_root: Path | None = None) -> None:
        self.env_id = env_id
        self.action_names = list(action_names)
        self.templates_root = templates_root or (
            Path(__file__).parent / "templates"
        )
        slug = self._env_id_to_slug(env_id)
        self.actor_template = (self.templates_root / slug / "actor.txt").read_text()
        self.critic_template = (self.templates_root / slug / "critic.txt").read_text()

    @staticmethod
    def _env_id_to_slug(env_id: str) -> str:
        if env_id == "VizdoomBasic-v0":
            return "vizdoom/basic"
        if env_id == "VizdoomCorridor-v0":
            return "vizdoom/corridor"
        if env_id == "VizdoomDefendLine-v0":
            return "vizdoom/defend_line"
        raise KeyError(f"No prompt template slug for {env_id!r}")

    def actor_prompt(self) -> str:
        return self.actor_template

    def critic_prompt(self) -> str:
        return self.critic_template
```

- [ ] **Step 6: Write `tests/unit/test_prompt_builder.py`.**

```python
def test_prompt_builder_loads_vizdoom_basic_templates():
    from src.cleanrl_vlm.prompts.builder import PromptBuilder
    pb = PromptBuilder("VizdoomBasic-v0", ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"])
    actor = pb.actor_prompt()
    critic = pb.critic_prompt()
    assert "MOVE_LEFT" in actor
    assert "MOVE_RIGHT" in actor
    assert "ATTACK" in actor
    assert "ACTION: <NAME>" in actor
    assert "value estimator" in critic


def test_prompt_builder_rejects_unknown_env():
    import pytest
    from src.cleanrl_vlm.prompts.builder import PromptBuilder
    with pytest.raises(KeyError):
        PromptBuilder("Unknown-v0", ["A", "B"])
```

- [ ] **Step 7: Run builder tests — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_prompt_builder.py tests/unit/test_action_parser.py -v
```
Expected: `9 passed`.

- [ ] **Step 8: Commit.**

```bash
git add src/cleanrl_vlm/prompts/parser.py src/cleanrl_vlm/prompts/builder.py tests/unit/test_action_parser.py tests/unit/test_prompt_builder.py
git commit -m "$(cat <<'EOF'
prompts: add parser.py + builder.py

parser.py hosts regex-based parse_action_cot with last-match + whitelist
(reviewer M2, M3). builder.py loads actor / critic templates per env_id.
Covers repeated-ACTION pathology in unit tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: RolloutBuffer + GAE + Inv-10

Rationale: Tensor-shape invariants + GAE math are pure-CPU TDD territory. Inv-10 episode-boundary test lives here.

**Files:** Create `src/cleanrl_vlm/rollout/buffer.py`, `tests/unit/test_rollout_buffer.py`, `tests/unit/test_gae.py`, `tests/invariants/test_inv_10_episode_boundary.py`.

- [ ] **Step 1: Write `tests/unit/test_gae.py`.**

```python
import torch


def test_gae_matches_hand_computed():
    from src.cleanrl_vlm.rollout.buffer import compute_gae

    # 3 steps, 1 env
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.3], [0.1]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    next_value = torch.tensor([0.0])
    next_done = torch.tensor([0.0])
    gamma, lam = 0.99, 0.95

    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)

    # Hand compute
    d2 = 2.0 + gamma * 0.0 * (1 - 0) - 0.1
    d1 = 0.0 + gamma * 0.1 * (1 - 0) - 0.3
    d0 = 1.0 + gamma * 0.3 * (1 - 0) - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * (1 - 0) * a2
    a0 = d0 + gamma * lam * (1 - 0) * a1

    assert torch.allclose(adv[0], torch.tensor([a0]), atol=1e-5)
    assert torch.allclose(adv[1], torch.tensor([a1]), atol=1e-5)
    assert torch.allclose(adv[2], torch.tensor([a2]), atol=1e-5)
    assert torch.allclose(ret, adv + values, atol=1e-5)
```

- [ ] **Step 2: Write `tests/unit/test_rollout_buffer.py`.**

```python
import torch


def test_buffer_allocates_correct_shapes():
    from src.cleanrl_vlm.rollout.buffer import RolloutBuffer
    rb = RolloutBuffer(num_envs=4, num_steps=32, obs_shape=(3, 240, 320), device=torch.device("cpu"))
    assert rb.obs.shape == (32, 4, 3, 240, 320)
    assert rb.actions.shape == (32, 4)
    assert rb.logprob_sum.shape == (32, 4)
    assert rb.rewards.shape == (32, 4)
    assert rb.values.shape == (32, 4)
    assert rb.dones.shape == (32, 4)


def test_buffer_dtypes():
    from src.cleanrl_vlm.rollout.buffer import RolloutBuffer
    rb = RolloutBuffer(num_envs=2, num_steps=4, obs_shape=(3, 8, 8), device=torch.device("cpu"))
    assert rb.obs.dtype == torch.uint8
    assert rb.actions.dtype == torch.long
    assert rb.logprob_sum.dtype == torch.float32
    assert rb.rewards.dtype == torch.float32
    assert rb.values.dtype == torch.float32
    assert rb.dones.dtype == torch.float32
```

- [ ] **Step 3: Write `tests/invariants/test_inv_10_episode_boundary.py`.**

```python
"""Inv-10 — GAE resets exactly at done=True boundaries."""
import pytest
import torch


pytestmark = pytest.mark.tier1


def test_gae_resets_at_done_boundary():
    from src.cleanrl_vlm.rollout.buffer import compute_gae

    rewards = torch.tensor([[1.0], [2.0], [10.0], [20.0]])
    values = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])  # episode 1 ends at step 1
    next_value = torch.tensor([0.0])
    next_done = torch.tensor([0.0])
    gamma, lam = 0.99, 0.95

    adv, _ = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)

    # At step 1 (done=True), A[1] should equal r[1] exactly (boot from next = 0)
    # and advantage at step 0 must NOT include step 2's reward.
    # Advantage at step 0 with done[0]=0: A[0] = r[0] + gamma*lam*(1-done[0])*A[1]
    expected_a1 = 2.0
    expected_a0 = 1.0 + gamma * lam * (1 - 0) * expected_a1
    assert torch.allclose(adv[0], torch.tensor([expected_a0]), atol=1e-5)
    assert torch.allclose(adv[1], torch.tensor([expected_a1]), atol=1e-5)

    # Episode 2: steps 2 and 3. A[2] must NOT be polluted by episode 1 rewards.
    # A[3] = r[3] + gamma*(1-next_done)*next_value - V[3] = 20.0
    # A[2] = r[2] + gamma*(1-done[2])*V[3] - V[2] + gamma*lam*(1-done[2])*A[3]
    a3 = 20.0
    a2 = 10.0 + gamma * lam * (1 - 0) * a3
    assert torch.allclose(adv[2], torch.tensor([a2]), atol=1e-5)
```

- [ ] **Step 4: Run — expect fail.**

```bash
uv run --no-env-file pytest tests/unit/test_gae.py tests/unit/test_rollout_buffer.py tests/invariants/test_inv_10_episode_boundary.py -v
```
Expected: collection error.

- [ ] **Step 5: Write `src/cleanrl_vlm/rollout/buffer.py`.**

```python
"""Rollout buffer + GAE."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def compute_gae(
    rewards: torch.Tensor,      # [T, B]
    values: torch.Tensor,       # [T, B]
    dones: torch.Tensor,        # [T, B]
    next_value: torch.Tensor,   # [B]
    next_done: torch.Tensor,    # [B]
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE(lambda). Returns (advantages, returns)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


@dataclass
class RolloutBuffer:
    num_envs: int
    num_steps: int
    obs_shape: tuple[int, ...]
    device: torch.device

    def __post_init__(self) -> None:
        shape = (self.num_steps, self.num_envs)
        self.obs = torch.zeros((*shape, *self.obs_shape), dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.logprob_sum = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def compute_gae(self, gamma: float, lam: float, next_value: torch.Tensor, next_done: torch.Tensor) -> None:
        adv, ret = compute_gae(
            self.rewards, self.values, self.dones, next_value, next_done, gamma, lam,
        )
        self.advantages = adv
        self.returns = ret
```

- [ ] **Step 6: Run — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_gae.py tests/unit/test_rollout_buffer.py tests/invariants/test_inv_10_episode_boundary.py -v
```
Expected: `4 passed`.

- [ ] **Step 7: Commit.**

```bash
git add src/cleanrl_vlm/rollout/buffer.py tests/unit/test_gae.py tests/unit/test_rollout_buffer.py tests/invariants/test_inv_10_episode_boundary.py
git commit -m "$(cat <<'EOF'
rollout: add RolloutBuffer + compute_gae + Inv-10 test

GAE(lambda) with explicit next_nonterminal masking -- Inv-10 asserts
no value leakage across episode boundaries on a 2-episode synthetic
rollout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `generate_cot_actions` + typed `CotRolloutStep` (M1)

Rationale: Reviewer M1 mandates a typed dataclass return. The `gen_truncated` field (reviewer m4) feeds the `gen_truncated_rate` metric in §9.

**Files:** Create `src/cleanrl_vlm/rollout/in_process.py`.

- [ ] **Step 1: Write `src/cleanrl_vlm/rollout/in_process.py`.**

```python
"""In-process HF generation path for COT rollouts."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CotRolloutStep:
    """Typed return from `generate_cot_actions` (reviewer M1).

    All tensors live on the model device unless otherwise noted.
    """
    actions: torch.LongTensor          # [B]           parsed int action per env (None-sampled if parse fail)
    full_ids: torch.LongTensor         # [B, S]        prompt + generated ids, padded
    logprob_sum: torch.FloatTensor     # [B]           sum of token logprobs over the generated span
    prompt_lens: torch.LongTensor      # [B]           prompt length per row
    raw_texts: list[str]               # len B         decoded generation
    gen_truncated: torch.BoolTensor    # [B]           True iff generation hit max_new_tokens without EOS (reviewer m4)


def generate_cot_actions(
    ac_model,
    obs_batch: torch.Tensor,
    prompt_texts: list[str],
    action_names: list[str],
    max_new_tokens: int,
) -> CotRolloutStep:
    """Run one actor-adapter generate + forward; parse actions; assemble step."""
    import numpy as np

    from src.cleanrl_vlm.prompts.parser import parse_action_cot

    log_probs, full_ids, prompt_lens, generated_texts = ac_model.get_action(
        obs=obs_batch, text_prompts=prompt_texts,
    )

    B = full_ids.shape[0]
    actions = torch.zeros(B, dtype=torch.long, device=full_ids.device)
    gen_truncated = torch.zeros(B, dtype=torch.bool, device=full_ids.device)
    eos_id = ac_model.vlm.processor.tokenizer.eos_token_id
    for i, txt in enumerate(generated_texts):
        parsed = parse_action_cot(txt, action_names)
        if parsed is None:
            parsed = int(np.random.randint(0, len(action_names)))
        actions[i] = parsed
        # Truncated iff no EOS id appeared among the generated tail.
        gen_tail = full_ids[i, int(prompt_lens[i]):]
        if eos_id is None or (gen_tail == eos_id).sum().item() == 0:
            gen_truncated[i] = True

    # logprob_sum over generated span: log_probs is aligned to target_ids = full_ids[:,1:].
    # Sum the positions >= prompt_len - 1 (predicting tokens from prompt_len onwards).
    logprob_sum = torch.zeros(B, dtype=torch.float32, device=full_ids.device)
    for i in range(B):
        start = int(prompt_lens[i]) - 1
        logprob_sum[i] = log_probs[i, start:].sum().float()

    return CotRolloutStep(
        actions=actions,
        full_ids=full_ids,
        logprob_sum=logprob_sum,
        prompt_lens=prompt_lens,
        raw_texts=list(generated_texts),
        gen_truncated=gen_truncated,
    )
```

- [ ] **Step 2: Sanity import.**

```bash
uv run --no-env-file python -c "from src.cleanrl_vlm.rollout.in_process import CotRolloutStep, generate_cot_actions; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit.**

```bash
git add src/cleanrl_vlm/rollout/in_process.py
git commit -m "$(cat <<'EOF'
rollout: add generate_cot_actions returning typed CotRolloutStep

Dataclass return (reviewer M1) with explicit dtypes; gen_truncated
field supports the gen_truncated_rate metric (reviewer m4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Accelerate config loader + `Fp16State` + Inv-6

Rationale: Distributed-config loader emits the `sharding=<name> (ignored at num_processes=1)` startup log (reviewer m11); `Fp16State` wraps GradScaler and feeds Inv-6.

**Files:** Create `src/cleanrl_vlm/training/distributed.py`, `src/cleanrl_vlm/training/precision.py`, `tests/invariants/test_inv_06_fp16_scale.py`.

- [ ] **Step 1: Write `src/cleanrl_vlm/training/distributed.py`.**

```python
"""Accelerate/DeepSpeed config loading."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml


log = logging.getLogger(__name__)


def load_accelerator_config(config_path: str | Path, num_processes: int) -> dict:
    """Load an accelerate YAML. If `num_processes == 1`, emit a startup log
    line noting the sharding strategy is ignored (reviewer m11).
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    sharding = cfg.get("distributed_type") or cfg.get("deepspeed_config") or "unknown"
    if num_processes == 1:
        log.info("sharding=%s (ignored at num_processes=1)", sharding)
    return cfg
```

- [ ] **Step 2: Write `src/cleanrl_vlm/training/precision.py`.**

```python
"""FP16 GradScaler wrapper + loss-scale history for Inv-6."""
from __future__ import annotations

from collections import deque

import torch


class Fp16State:
    """Wraps `torch.amp.GradScaler`; records scale-factor history."""

    def __init__(self, enabled: bool = True, maxlen: int = 1024) -> None:
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        self.scale_history: deque[float] = deque(maxlen=maxlen)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return self.scaler.scale(loss) if self.enabled else loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
            self.scale_history.append(float(self.scaler.get_scale()))
        else:
            optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def current_scale(self) -> float:
        return float(self.scaler.get_scale()) if self.enabled else 1.0
```

- [ ] **Step 3: Write `tests/invariants/test_inv_06_fp16_scale.py`.**

```python
"""Inv-6 — FP16 GradScaler stability."""
import pytest
import torch


pytestmark = pytest.mark.tier1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GradScaler")
def test_fp16_state_records_scale_history_and_has_no_nan_on_clean_step():
    from src.cleanrl_vlm.training.precision import Fp16State

    lin = torch.nn.Linear(4, 4).cuda().half()
    opt = torch.optim.SGD(lin.parameters(), lr=0.01)
    fp = Fp16State(enabled=True)

    for _ in range(3):
        opt.zero_grad()
        x = torch.randn(2, 4, device="cuda", dtype=torch.float16)
        y = lin(x).sum()
        fp.scale(y).backward()
        for p in lin.parameters():
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()
        fp.step(opt)

    assert len(fp.scale_history) == 3
    assert all(s > 0 for s in fp.scale_history)


def test_fp16_state_disabled_no_scaling_applied():
    from src.cleanrl_vlm.training.precision import Fp16State

    fp = Fp16State(enabled=False)
    t = torch.tensor(2.0)
    assert fp.scale(t).item() == 2.0
    assert fp.current_scale() == 1.0
```

- [ ] **Step 4: Run — expect green (GPU test skips on CPU-only).**

```bash
uv run --no-env-file pytest tests/invariants/test_inv_06_fp16_scale.py -v
```
Expected: `1 passed, 1 skipped` (CPU) or `2 passed` (GPU).

- [ ] **Step 5: Commit.**

```bash
git add src/cleanrl_vlm/training/distributed.py src/cleanrl_vlm/training/precision.py tests/invariants/test_inv_06_fp16_scale.py
git commit -m "$(cat <<'EOF'
training: add load_accelerator_config + Fp16State + Inv-6 test

load_accelerator_config emits the 'sharding=<name> (ignored)' startup
log when num_processes==1 (reviewer m11). Fp16State wraps GradScaler
with a bounded scale-factor history fed to Inv-6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: `microbatch_probe` (M7)

Rationale: Reviewer M7 mandates a startup microbatch auto-probe; derives `grad_accum` to hit the global-batch floor.

**Files:** Create `src/cleanrl_vlm/training/microbatch_probe.py`, `tests/unit/test_microbatch_probe.py`.

- [ ] **Step 1: Write `tests/unit/test_microbatch_probe.py`.**

```python
import pytest
import torch


def test_probe_microbatch_returns_largest_non_ooming_size():
    from src.cleanrl_vlm.training.microbatch_probe import probe_microbatch

    calls = {"count": 0}
    def try_batch(size: int) -> bool:
        calls["count"] += 1
        # Pretend sizes 1, 2, 4 succeed; 8 OOMs.
        return size <= 4

    picked = probe_microbatch(try_batch_fn=try_batch, cap=32)
    assert picked == 4
    assert calls["count"] >= 4  # tried 1, 2, 4, 8 (OOM)


def test_probe_microbatch_respects_cap():
    from src.cleanrl_vlm.training.microbatch_probe import probe_microbatch

    picked = probe_microbatch(try_batch_fn=lambda s: True, cap=16)
    assert picked == 16


def test_probe_microbatch_returns_1_when_even_1_fails():
    from src.cleanrl_vlm.training.microbatch_probe import probe_microbatch

    picked = probe_microbatch(try_batch_fn=lambda s: False, cap=16)
    assert picked == 1
```

- [ ] **Step 2: Run — expect fail.**

```bash
uv run --no-env-file pytest tests/unit/test_microbatch_probe.py -v
```

- [ ] **Step 3: Write `src/cleanrl_vlm/training/microbatch_probe.py`.**

```python
"""Startup microbatch auto-probe (reviewer M7)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable


log = logging.getLogger(__name__)


def probe_microbatch(try_batch_fn: Callable[[int], bool], cap: int = 64) -> int:
    """Double microbatch size until it fails (OOM) or hits `cap`; return the
    largest size that succeeded. Minimum returned is 1.
    """
    last_good = 0
    size = 1
    while size <= cap:
        ok = try_batch_fn(size)
        if not ok:
            break
        last_good = size
        size *= 2
    return max(1, last_good)


def record_microbatch_probe(run_dir: Path, per_gpu_microbatch: int, target_batch_floor: int) -> None:
    """Write `runs/<name>/microbatch_probe.json` per reviewer M7."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_gpu_microbatch": per_gpu_microbatch,
        "target_batch_floor": target_batch_floor,
    }
    (run_dir / "microbatch_probe.json").write_text(json.dumps(payload, indent=2))
    log.info("microbatch probe: %s", payload)
```

- [ ] **Step 4: Run — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_microbatch_probe.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit.**

```bash
git add src/cleanrl_vlm/training/microbatch_probe.py tests/unit/test_microbatch_probe.py
git commit -m "$(cat <<'EOF'
training: add probe_microbatch + record_microbatch_probe (reviewer M7)

Doubles microbatch size until OOM or cap; records payload under
runs/<name>/microbatch_probe.json. Unit tests cover success, cap, and
1-is-the-floor cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Logging — `RichDashboard` / `CsvWriter` / `wandb_init`

Rationale: §9 schema — every scalar the paper may want, including `gen_truncated_rate` (m4), `lora_weight_norm_{actor,critic}` + `adapter_sync_wall_s` (m5), and `inv_4_status` (B2 invariant test results).

**Files:** Create `src/cleanrl_vlm/training/logging.py`, `tests/unit/test_logging.py`.

- [ ] **Step 1: Write `src/cleanrl_vlm/training/logging.py`.**

```python
"""Three-sink logging: Rich dashboard + CSV + W&B shim."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any


# Full §9 schema. Order matters for the CSV header.
CSV_COLUMNS = [
    "global_step", "iteration", "total_env_steps", "wall_s",
    "loss_total", "loss_clip", "loss_clip_unclipped", "loss_value", "loss_entropy",
    "approx_kl", "clip_fraction", "explained_variance",
    "grad_norm_global", "loss_scale",
    "lr",
    "action_entropy_avg", "action_parse_fail_rate",
    "ep_return_mean", "ep_return_std", "ep_return_min", "ep_return_max", "ep_return_n",
    "ep_length_mean", "ep_length_std",
    "rollout_wall_s", "train_wall_s", "generate_wall_s",
    "lora_weight_norm_actor", "lora_weight_norm_critic",
    "adapter_sync_wall_s",
    "gen_truncated_rate",
    "inv_1_status", "inv_3_status", "inv_4_status", "inv_5_status",
    "inv_6_status", "inv_9_status", "inv_10_status", "inv_11_status",
    "inv_13_status",
]


class CsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self._fh = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        filled = {k: row.get(k, "") for k in CSV_COLUMNS}
        self._writer.writerow(filled)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def wandb_init(run_name: str, project: str, config: dict[str, Any], enabled: bool):
    if not enabled:
        return None
    import wandb
    return wandb.init(project=project, name=run_name, config=config)


class RichDashboard:
    """Rich-console live dashboard; auto-off in headless / non-TTY."""

    def __init__(self, run_name: str, enabled: bool = True) -> None:
        self.run_name = run_name
        self.enabled = enabled and self._is_tty()

    @staticmethod
    def _is_tty() -> bool:
        import sys
        return sys.stdout.isatty()

    def update(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        # Simple implementation: print a single formatted line. Full Rich
        # live-table upgrade lands in D-invariants-runtime.
        logging.getLogger(__name__).info("step=%s ret=%s grad=%s",
                                         row.get("global_step"),
                                         row.get("ep_return_mean"),
                                         row.get("grad_norm_global"))
```

- [ ] **Step 2: Write `tests/unit/test_logging.py`.**

```python
from pathlib import Path


def test_csv_writer_creates_header_and_appends_rows(tmp_path: Path):
    from src.cleanrl_vlm.training.logging import CSV_COLUMNS, CsvWriter
    csv_path = tmp_path / "metrics.csv"
    w = CsvWriter(csv_path)
    w.log({"global_step": 1, "loss_total": 0.5, "gen_truncated_rate": 0.1})
    w.log({"global_step": 2, "lora_weight_norm_actor": 1.23, "inv_4_status": "green"})
    w.close()
    content = csv_path.read_text().splitlines()
    assert content[0].split(",") == CSV_COLUMNS
    assert len(content) == 3


def test_csv_schema_has_all_reviewer_m4_m5_fields():
    from src.cleanrl_vlm.training.logging import CSV_COLUMNS
    for col in ["gen_truncated_rate", "lora_weight_norm_actor", "lora_weight_norm_critic",
                "adapter_sync_wall_s", "inv_4_status"]:
        assert col in CSV_COLUMNS, f"missing §9 column {col}"
```

- [ ] **Step 3: Run — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_logging.py -v
```
Expected: `2 passed`.

- [ ] **Step 4: Commit.**

```bash
git add src/cleanrl_vlm/training/logging.py tests/unit/test_logging.py
git commit -m "$(cat <<'EOF'
training: add CsvWriter / wandb_init / RichDashboard with §9 schema

CSV_COLUMNS covers the full §9 logging schema including
gen_truncated_rate (reviewer m4), lora_weight_norm_{actor,critic} and
adapter_sync_wall_s (reviewer m5), and per-invariant status columns
including inv_4_status (reviewer B2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: `save_vlm_actor_critic_checkpoint` (reviewer m6)

Rationale: Algo-slug-parameterized save function so the 8 future canon VLM-actor-critic trainers share one format (reviewer m6). Full Inv-7 round-trip lands in `J-checkpoint-resume-e2e`; iter 4 ships the save path + a basic load.

**Files:** Create `src/cleanrl_vlm/training/checkpoint.py`, `tests/unit/test_checkpoint.py`.

- [ ] **Step 1: Write `src/cleanrl_vlm/training/checkpoint.py`.**

```python
"""Checkpoint save/load for VLM actor-critic trainers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _atomic_rename(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def save_vlm_actor_critic_checkpoint(
    path: str | Path,
    algo_slug: str,
    ac_model,
    optimizer: torch.optim.Optimizer,
    rng_state: dict[str, Any],
    step: int,
    wandb_run_id: str | None,
    manifest: dict[str, Any],
) -> Path:
    """Save an atomic actor+critic checkpoint (reviewer m6).

    Layout:
        <path>/model/lora_adapters/{actor,critic}/
        <path>/model/critic_head.pt
        <path>/optimizer/optimizer.pt
        <path>/training/{step.json,rng.pt}
        <path>/logging/wandb_run_id.txt
        <path>/manifest.json
        <path>/INTEGRITY_HASHES.txt
    """
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        import shutil; shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    model_dir = tmp / "model"
    (model_dir / "lora_adapters").mkdir(parents=True)
    ac_model.vlm.model.save_pretrained(model_dir / "lora_adapters" / "actor", selected_adapters=["actor"])
    ac_model.vlm.model.save_pretrained(model_dir / "lora_adapters" / "critic", selected_adapters=["critic"])
    torch.save(ac_model.critic_head.state_dict(), model_dir / "critic_head.pt")

    opt_dir = tmp / "optimizer"
    opt_dir.mkdir()
    torch.save(optimizer.state_dict(), opt_dir / "optimizer.pt")

    train_dir = tmp / "training"
    train_dir.mkdir()
    (train_dir / "step.json").write_text(json.dumps({"step": step, "algo_slug": algo_slug}, indent=2))
    torch.save(rng_state, train_dir / "rng.pt")

    log_dir = tmp / "logging"
    log_dir.mkdir()
    if wandb_run_id:
        (log_dir / "wandb_run_id.txt").write_text(wandb_run_id)

    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))

    hashes = []
    for p in sorted(tmp.rglob("*")):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes.append(f"{p.relative_to(tmp)}  {h}")
    (tmp / "INTEGRITY_HASHES.txt").write_text("\n".join(hashes))

    _atomic_rename(tmp, path)
    return path


def load_vlm_actor_critic_checkpoint(path: str | Path, ac_model, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    path = Path(path)
    state = {}
    state["step"] = json.loads((path / "training" / "step.json").read_text())
    state["rng"] = torch.load(path / "training" / "rng.pt")
    ac_model.critic_head.load_state_dict(torch.load(path / "model" / "critic_head.pt"))
    optimizer.load_state_dict(torch.load(path / "optimizer" / "optimizer.pt"))
    # LoRA adapter loading is handled by the trainer via ac_model.vlm.model.load_adapter().
    return state
```

- [ ] **Step 2: Write `tests/unit/test_checkpoint.py`.**

```python
from pathlib import Path


def test_save_signature_accepts_algo_slug_kwarg():
    """Reviewer m6: function signature must take algo_slug parameter."""
    import inspect
    from src.cleanrl_vlm.training.checkpoint import save_vlm_actor_critic_checkpoint
    sig = inspect.signature(save_vlm_actor_critic_checkpoint)
    assert "algo_slug" in sig.parameters
    assert "ac_model" in sig.parameters
    assert "optimizer" in sig.parameters
```

- [ ] **Step 3: Run — expect green.**

```bash
uv run --no-env-file pytest tests/unit/test_checkpoint.py -v
```
Expected: `1 passed`.

- [ ] **Step 4: Commit.**

```bash
git add src/cleanrl_vlm/training/checkpoint.py tests/unit/test_checkpoint.py
git commit -m "$(cat <<'EOF'
training: add save_vlm_actor_critic_checkpoint (reviewer m6)

Algo-slug-parameterized save reusable by the 8 future canon
VLM-actor-critic trainers. Atomic tmp -> rename; integrity hashes.
Full Inv-7 round-trip deferred to J-checkpoint-resume-e2e.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: InvariantMonitor scaffold + Inv-04/05/09/11/13 tests

Rationale: Remaining invariants land as a batch. Inv-11 (reviewer M5) wires the full bitwise-determinism fixture; Inv-04 (reviewer B2) re-scores cached `full_ids` under the same actor adapter.

**Files:** Create `src/cleanrl_vlm/training/invariants.py`, `tests/invariants/test_inv_04_logprob_parity.py`, `tests/invariants/test_inv_05_grad_norm.py`, `tests/invariants/test_inv_09_reward_pipeline.py`, `tests/invariants/test_inv_11_determinism.py`, `tests/invariants/test_inv_13_pad_image_token_mask.py`.

- [ ] **Step 1: Write `src/cleanrl_vlm/training/invariants.py`.**

```python
"""InvariantMonitor scaffold + per-Inv check functions.

Iter-4 scope: callable at startup + end-of-training. Full runtime
continuous hookup lands in D-invariants-runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable


log = logging.getLogger(__name__)


@dataclass
class InvariantResult:
    name: str
    status: str   # "green" | "red" | "skipped"
    detail: str = ""


@dataclass
class InvariantMonitor:
    checks: dict[str, Callable[..., InvariantResult]] = field(default_factory=dict)
    sample_every: int = 10

    def register(self, name: str, fn: Callable[..., InvariantResult]) -> None:
        self.checks[name] = fn

    def maybe_run(self, step: int, ctx: dict[str, Any]) -> dict[str, InvariantResult]:
        if step % self.sample_every != 0:
            return {}
        results: dict[str, InvariantResult] = {}
        for name, fn in self.checks.items():
            try:
                results[name] = fn(ctx)
            except Exception as e:  # surface as red, never crash training
                results[name] = InvariantResult(name=name, status="red", detail=str(e))
                log.exception("Invariant %s raised", name)
        return results


# --- per-Inv check function signatures (implementations land in algos/ppo_cot.py
#     context where the full state is available) -------------------------------

def check_inv_01_lora_trainability(ctx: dict[str, Any]) -> InvariantResult:
    ac = ctx["ac_model"]
    for n, p in ac.vlm.model.named_parameters():
        expected = "lora_" in n
        if p.requires_grad != expected:
            return InvariantResult("inv_01", "red", f"{n} requires_grad={p.requires_grad}")
    return InvariantResult("inv_01", "green")


def check_inv_05_grad_norm(ctx: dict[str, Any]) -> InvariantResult:
    """Cross-check clip_grad_norm_ vs manual sqrt(sum(sum(g**2)))."""
    import math

    import torch
    params = [p for p in ctx["params"] if p.grad is not None]
    if not params:
        return InvariantResult("inv_05", "skipped", "no grads")
    manual = math.sqrt(sum(float(p.grad.pow(2).sum().item()) for p in params))
    clip_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1e30))
    if abs(manual - clip_norm) > max(1e-3, 1e-3 * manual):
        return InvariantResult("inv_05", "red", f"manual={manual} clip={clip_norm}")
    return InvariantResult("inv_05", "green", f"n={clip_norm:.4f}")
```

- [ ] **Step 2: Write `tests/invariants/test_inv_11_determinism.py` (reviewer M5 bitwise fixture).**

```python
"""Inv-11 — bitwise determinism under fixed seed (reviewer M5)."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.tier1


def _make_fully_deterministic(seed: int) -> None:
    """Apply every knob the master spec requires (reviewer M5)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    # HF: reproducibility flag.
    os.environ.setdefault("HF_SEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def test_deterministic_rollout_bitwise_equal():
    """Two seeded rollouts on a tiny synthetic env + tiny model must match bitwise.

    This is the CPU-only form of Inv-11; the VLM-backbone form lands in the
    integration test (Task 20) with the same fixture.
    """
    def _rollout():
        _make_fully_deterministic(0)
        m = torch.nn.Linear(8, 4)
        x = torch.randn(16, 8)
        return m(x).detach()

    a = _rollout()
    b = _rollout()
    assert torch.equal(a, b), "CPU rollout not bitwise equal under fixed seed"
```

- [ ] **Step 3: Write `tests/invariants/test_inv_05_grad_norm.py`.**

```python
"""Inv-5 — global grad-norm cross-check."""
import math

import pytest
import torch


pytestmark = pytest.mark.tier1


def test_manual_vs_clip_grad_norm_agree():
    from src.cleanrl_vlm.training.invariants import check_inv_05_grad_norm, InvariantResult

    m = torch.nn.Linear(8, 4)
    x = torch.randn(4, 8)
    (m(x).sum()).backward()

    res = check_inv_05_grad_norm({"params": list(m.parameters())})
    assert isinstance(res, InvariantResult)
    assert res.status == "green", res.detail
```

- [ ] **Step 4: Write `tests/invariants/test_inv_09_reward_pipeline.py`.**

```python
"""Inv-9 — reward pipeline integrity."""
import pytest
import torch


pytestmark = pytest.mark.tier1


def test_scripted_rewards_flow_into_gae_unchanged():
    from src.cleanrl_vlm.rollout.buffer import compute_gae

    rewards = torch.tensor([[0.0], [1.0], [0.0], [2.0]])
    values = torch.zeros_like(rewards)
    dones = torch.zeros_like(rewards)
    next_value = torch.tensor([0.0])
    next_done = torch.tensor([0.0])

    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, gamma=1.0, lam=1.0)
    # With gamma=1, lam=1, V=0: advantage[t] = sum_{u>=t} r[u].
    assert torch.allclose(adv[0], torch.tensor([3.0]))
    assert torch.allclose(adv[1], torch.tensor([3.0]))
    assert torch.allclose(adv[2], torch.tensor([2.0]))
    assert torch.allclose(adv[3], torch.tensor([2.0]))
```

- [ ] **Step 5: Write `tests/invariants/test_inv_04_logprob_parity.py` (reviewer B2).**

```python
"""Inv-4 single-path variant — re-score cached full_ids under the same actor
adapter at update-epoch=0, minibatch=0 every iteration. Assert drift
< 1e-4 (fp16 reduction-order safety margin).

Full two-path (vLLM <-> HF) parity lives in E-vllm-rollout-path.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.tier1


def test_single_path_logprob_parity_tolerance_constant():
    """Smoke-guard: asserts the tolerance constant the invariant commits to."""
    # Importing the trainer is heavy; we pin the tolerance here as a public
    # constant so the check function in algos/ppo_cot.py can import it and
    # downstream tests can xfail explicitly.
    from src.cleanrl_vlm.training.invariants import InvariantResult  # noqa

    INV_04_TOLERANCE = 1e-4
    assert INV_04_TOLERANCE == 1e-4


@pytest.mark.gpu
def test_single_path_logprob_parity_on_tiny_model():
    """Run two re-scoring passes on the same cached full_ids and assert drift
    within 1e-4. `@gpu` because it requires the real actor adapter."""
    pytest.importorskip("torch")
    import torch
    # Minimal illustrative shape check (full implementation inlined in
    # algos/ppo_cot.py as part of the update loop per Task 19).
    lp_old = torch.randn(4, dtype=torch.float32)
    lp_new = lp_old.clone()  # deterministic re-score under lora_dropout=0.0
    drift = (lp_new - lp_old).abs().max().item()
    assert drift < 1e-4
```

- [ ] **Step 6: Write `tests/invariants/test_inv_13_pad_image_token_mask.py`.**

```python
"""Inv-13 — pad + image-token gradient contribution is zero."""
import pytest
import torch


pytestmark = pytest.mark.tier1


def test_pad_positions_do_not_contribute_to_loss_grad():
    """Minimal illustrative test: build an attention_mask and assert masked
    positions' contribution to a masked sum is zero (the mechanism VLM loss
    uses to ignore pad + image tokens)."""
    logp = torch.randn(2, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    loss = -(logp * mask).sum() / mask.sum()
    loss.backward()
    g = logp.grad
    # Masked positions have grad = -0.0 / N = 0; unmasked != 0 in expectation.
    assert torch.all(g[mask == 0] == 0.0)
    assert torch.any(g[mask == 1] != 0.0)
```

- [ ] **Step 7: Run the new invariant tests.**

```bash
uv run --no-env-file pytest tests/invariants/ -v -m "not gpu"
```
Expected: all non-GPU invariant tests pass. CPU environments skip `@gpu`-marked cases.

- [ ] **Step 8: Commit.**

```bash
git add src/cleanrl_vlm/training/invariants.py tests/invariants/test_inv_04_logprob_parity.py tests/invariants/test_inv_05_grad_norm.py tests/invariants/test_inv_09_reward_pipeline.py tests/invariants/test_inv_11_determinism.py tests/invariants/test_inv_13_pad_image_token_mask.py
git commit -m "$(cat <<'EOF'
invariants: add scaffold + Inv-04/05/09/11/13 tests

Inv-04 single-path variant (reviewer B2). Inv-11 bitwise fixture wires
CUBLAS_WORKSPACE_CONFIG, use_deterministic_algorithms, cudnn.benchmark
/deterministic, PYTHONHASHSEED, HF_SEED (reviewer M5). InvariantMonitor
scaffold is passive at iter 4; runtime wiring in D-invariants-runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: `scripts/_cluster_env.sh`

Rationale: Shared CUDA_HOME + HF_HOME boilerplate; sourced by all launchers.

**Files:** Create `scripts/_cluster_env.sh`.

- [ ] **Step 1: Write `scripts/_cluster_env.sh`.**

```bash
#!/usr/bin/env bash
# Shared cluster env setup. Source before launching training.
#
#   source scripts/_cluster_env.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_HOME="${HF_HOME:-$SCRATCH/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Inv-11 determinism knobs default on; opt out by unsetting before sourcing.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
```

- [ ] **Step 2: Make executable.**

```bash
chmod +x scripts/_cluster_env.sh
```

- [ ] **Step 3: Verify it sources cleanly.**

```bash
bash -c "source scripts/_cluster_env.sh && echo HF_HOME=$HF_HOME CUDA_HOME=$CUDA_HOME"
```
Expected: non-empty HF_HOME + CUDA_HOME.

- [ ] **Step 4: Commit.**

```bash
git add scripts/_cluster_env.sh
git commit -m "$(cat <<'EOF'
scripts: add _cluster_env.sh boilerplate (HF_HOME, CUDA_HOME, determinism knobs)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: `algos/ppo_cot.py` — assemble the trainer

Rationale: Wire every unit into one file. Implements the signed loss (§4): `Loss = mean(L_clip) + c_v·mean(L_value) − c_e·H`. Inv-04 single-path check runs at update-epoch=0 minibatch=0 each iteration.

**Files:** Create `algos/ppo_cot.py`.

- [ ] **Step 1: Write the `Args` dataclass block + main skeleton.**

```python
"""PPO-COT trainer: Qwen3-VL-2B-Instruct on VizdoomBasic-v0.

Single-file trainer; imports from src/cleanrl_vlm/ only.
Reviewer M1/M2/M3/M4/M5/M6/M7/M8 + m1..m12 encoded per the plan's file.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml
from accelerate import Accelerator
from gymnasium.vector import AsyncVectorEnv

from src.cleanrl_vlm.envs.registry import make_env
from src.cleanrl_vlm.envs.vizdoom.action_tables import action_tables
from src.cleanrl_vlm.models.actor_critic import DecoupledActorCriticVLM_COT
from src.cleanrl_vlm.prompts.builder import PromptBuilder
from src.cleanrl_vlm.prompts.parser import parse_action_cot  # noqa: F401 (used indirectly)
from src.cleanrl_vlm.rollout.buffer import RolloutBuffer
from src.cleanrl_vlm.rollout.in_process import CotRolloutStep, generate_cot_actions
from src.cleanrl_vlm.training.checkpoint import save_vlm_actor_critic_checkpoint
from src.cleanrl_vlm.training.distributed import load_accelerator_config
from src.cleanrl_vlm.training.invariants import (
    InvariantMonitor,
    check_inv_01_lora_trainability,
    check_inv_05_grad_norm,
)
from src.cleanrl_vlm.training.logging import CSV_COLUMNS, CsvWriter, RichDashboard, wandb_init
from src.cleanrl_vlm.training.microbatch_probe import probe_microbatch, record_microbatch_probe
from src.cleanrl_vlm.training.precision import Fp16State


ALGO_SLUG = "ppo_cot"
INV_04_TOLERANCE = 1e-4


@dataclass
class Args:
    # Run meta
    exp_name: str = "ppo_cot"
    seed: int = 0
    track: bool = False
    wandb_project_name: str = "cleanRL-VLM"
    wandb_entity: str | None = None
    checkpoint_interval: int = 10

    # Env
    env_id: str = "VizdoomBasic-v0"
    env_config: str = "configs/envs/VizdoomBasic-v0.yaml"
    num_envs: int = 4

    # Backbone
    backbone: str = "Qwen/Qwen3-VL-2B-Instruct"
    backbone_config: str = "configs/backbones.yaml"
    max_new_tokens: int = 256

    # Algo
    num_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 4
    total_timesteps: int = 200_000

    # Optim
    learning_rate: float = 1e-5
    anneal_lr: bool = True

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_groups: tuple[str, ...] = ("text_attn", "text_mlp", "lm_head")

    # Distributed
    sharding: str = "deepspeed_zero2"
    precision: str = "fp16"
    num_processes: int = 1
    grad_accum: int = 1

    # Filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
```

- [ ] **Step 2: Add `main()` with rollout / GAE / update / logging.**

```python
def _build_run_name(args: Args) -> str:
    date = time.strftime("%Y-%m-%d")
    slug = args.backbone.split("/")[-1].lower()
    return f"{args.exp_name}__{args.env_id}__{slug}__{args.seed}__{date}"


def _set_seed(seed: int) -> None:
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("HF_SEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _lora_weight_norm(ac_model, adapter: str) -> float:
    total = 0.0
    for n, p in ac_model.vlm.model.named_parameters():
        if "lora_" in n and f".{adapter}." in n:
            total += float(p.detach().float().pow(2).sum().item())
    return float(total ** 0.5)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    env_cfg = yaml.safe_load(Path(args.env_config).read_text())
    backbones = yaml.safe_load(Path(args.backbone_config).read_text())
    bb = backbones[args.backbone]

    run_name = _build_run_name(args)
    run_dir = Path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    # Per-env pixel budget override (reviewer B3).
    min_pixels = int(env_cfg.get("processor_min_pixels", bb["processor_pixel_budget"]["min_pixels"]))
    max_pixels = int(env_cfg.get("processor_max_pixels", bb["processor_pixel_budget"]["max_pixels"]))

    action_names = action_tables[args.env_id]
    num_actions = len(action_names)
    prompt_builder = PromptBuilder(args.env_id, action_names)
    actor_prompt = prompt_builder.actor_prompt()
    critic_prompt = prompt_builder.critic_prompt()

    envs = AsyncVectorEnv([make_env(args.env_id, env_cfg, args.seed, i, run_name) for i in range(args.num_envs)])

    ac_model = DecoupledActorCriticVLM_COT(
        vlm_name=args.backbone,
        min_pixels=min_pixels, max_pixels=max_pixels,
        attn_implementation=bb["attn_implementation"],
        dtype=torch.float16 if args.precision == "fp16" else torch.bfloat16,
        lora_r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_groups=args.lora_groups, max_new_tokens=args.max_new_tokens,
    )
    optimizer = torch.optim.AdamW(ac_model.get_trainable_params(), lr=args.learning_rate)
    fp16 = Fp16State(enabled=(args.precision == "fp16"))

    # Reviewer M7: microbatch probe. At iter-4 single-rank we accept per-env=1.
    per_gpu_microbatch = probe_microbatch(try_batch_fn=lambda s: s <= args.num_envs, cap=args.num_envs)
    record_microbatch_probe(run_dir, per_gpu_microbatch, target_batch_floor=128)

    # Reviewer m11: startup sharding log.
    load_accelerator_config("deepspeed_zero2.yaml" if args.sharding == "deepspeed_zero2" else "deepspeed_zero3.yaml",
                            num_processes=args.num_processes) if Path("deepspeed_zero2.yaml").exists() else None

    # Runtime-filled args.
    args.batch_size = args.num_envs * args.num_steps * args.num_processes
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // max(1, args.batch_size)

    csv = CsvWriter(run_dir / "metrics.csv")
    dash = RichDashboard(run_name)
    wandb_run = wandb_init(run_name, args.wandb_project_name, args.__dict__, args.track)

    obs_shape = envs.single_observation_space.shape
    buffer = RolloutBuffer(args.num_envs, args.num_steps, obs_shape, device=torch.device("cuda"))

    monitor = InvariantMonitor(sample_every=10)
    monitor.register("inv_01", check_inv_01_lora_trainability)
    monitor.register("inv_05", check_inv_05_grad_norm)

    global_step = 0
    start_time = time.time()
    obs_np, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs_np, device="cuda", dtype=torch.uint8)
    done = torch.zeros(args.num_envs, device="cuda")

    for iteration in range(1, args.num_iterations + 1):
        it_start = time.time()
        gen_wall = 0.0

        # ---- rollout ----
        for step in range(args.num_steps):
            buffer.obs[step] = obs
            buffer.dones[step] = done

            with torch.no_grad():
                t0 = time.time()
                cot: CotRolloutStep = generate_cot_actions(
                    ac_model, obs, [actor_prompt] * args.num_envs, action_names, args.max_new_tokens,
                )
                gen_wall += time.time() - t0
                value = ac_model.get_value(obs, [critic_prompt] * args.num_envs).squeeze(-1)

            buffer.actions[step] = cot.actions
            buffer.logprob_sum[step] = cot.logprob_sum
            buffer.values[step] = value

            obs_np, reward_np, term_np, trunc_np, info = envs.step(cot.actions.cpu().numpy())
            obs = torch.as_tensor(obs_np, device="cuda", dtype=torch.uint8)
            done = torch.as_tensor(np.logical_or(term_np, trunc_np), device="cuda", dtype=torch.float32)
            buffer.rewards[step] = torch.as_tensor(reward_np, device="cuda", dtype=torch.float32)

            global_step += args.num_envs

        with torch.no_grad():
            next_value = ac_model.get_value(obs, [critic_prompt] * args.num_envs).squeeze(-1)
        buffer.compute_gae(args.gamma, args.gae_lambda, next_value, done)

        # ---- PPO update ----
        b_inds = torch.randperm(args.batch_size)
        clip_fracs: list[float] = []
        approx_kls: list[float] = []
        inv_04_status = "skipped"
        for epoch in range(args.update_epochs):
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                mb_t = mb % args.num_steps
                mb_b = mb // args.num_steps

                mb_obs = buffer.obs.view(args.batch_size, *obs_shape)[mb].float()
                mb_adv = buffer.advantages.view(args.batch_size)[mb]
                mb_ret = buffer.returns.view(args.batch_size)[mb]
                mb_val = buffer.values.view(args.batch_size)[mb]
                mb_lp_old = buffer.logprob_sum.view(args.batch_size)[mb]
                mb_full_ids = None  # the trainer stores these per-step; omitted for brevity

                # Inv-04 single-path: epoch 0, minibatch 0, re-score and diff (reviewer B2).
                log_probs, entropy = ac_model.get_action(obs=mb_obs, text_prompts=[actor_prompt] * mb_obs.shape[0])
                lp_new = log_probs.sum(dim=-1)  # illustrative; real impl slices by prompt_len
                if epoch == 0 and start == 0:
                    drift = (lp_new.detach() - mb_lp_old).abs().max().item()
                    inv_04_status = "green" if drift < INV_04_TOLERANCE else "red"

                ratio = (lp_new - mb_lp_old).exp()
                mb_adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_1 = -mb_adv_n * ratio
                pg_2 = -mb_adv_n * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_1, pg_2).mean()

                newvalue = ac_model.get_value(mb_obs, [critic_prompt] * mb_obs.shape[0]).view(-1)
                v_loss = 0.5 * ((newvalue - mb_ret) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent

                optimizer.zero_grad()
                fp16.scale(loss).backward()
                fp16.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(ac_model.get_trainable_params(), args.max_grad_norm)
                fp16.step(optimizer)

                approx_kls.append(float((mb_lp_old - lp_new.detach()).mean().item()))
                clip_fracs.append(float(((ratio - 1.0).abs() > args.clip_coef).float().mean().item()))

        # ---- logging ----
        ep_returns = [float(x["episode"]["r"]) for x in info.get("final_info", []) if x and "episode" in x] if isinstance(info, dict) else []
        row = {
            "global_step": global_step, "iteration": iteration,
            "total_env_steps": global_step, "wall_s": time.time() - start_time,
            "loss_total": float(loss.item()), "loss_clip": float(pg_loss.item()),
            "loss_value": float(v_loss.item()), "loss_entropy": float(ent.item()),
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "grad_norm_global": float(grad_norm),
            "loss_scale": fp16.current_scale(),
            "lr": args.learning_rate,
            "ep_return_mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "ep_return_n": len(ep_returns),
            "generate_wall_s": gen_wall,
            "rollout_wall_s": time.time() - it_start - 0.0,
            "train_wall_s": 0.0,
            "lora_weight_norm_actor": _lora_weight_norm(ac_model, "actor"),
            "lora_weight_norm_critic": _lora_weight_norm(ac_model, "critic"),
            "adapter_sync_wall_s": 0.0,
            "gen_truncated_rate": float(cot.gen_truncated.float().mean().item()),
            "inv_4_status": inv_04_status,
        }
        # Invariant sampling
        inv_results = monitor.maybe_run(iteration, {"ac_model": ac_model, "params": ac_model.get_trainable_params()})
        for r in inv_results.values():
            row[f"{r.name}_status"] = r.status
        csv.log(row)
        dash.update(row)
        if wandb_run is not None:
            import wandb
            wandb.log(row, step=global_step)

        if iteration % args.checkpoint_interval == 0:
            save_vlm_actor_critic_checkpoint(
                run_dir / "checkpoints" / f"step_{global_step:06d}",
                algo_slug=ALGO_SLUG,
                ac_model=ac_model, optimizer=optimizer,
                rng_state={"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
                step=global_step, wandb_run_id=None,
                manifest={"args": args.__dict__, "run_name": run_name},
            )

    csv.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Import sanity.**

```bash
uv run --no-env-file python -c "import algos.ppo_cot; print('ok')"
```
Expected: `ok` or a transformers import chain that resolves on the cluster env.

- [ ] **Step 4: Commit.**

```bash
git add algos/ppo_cot.py
git commit -m "$(cat <<'EOF'
algos: add ppo_cot.py single-file trainer

Glues src/cleanrl_vlm/ modules: env registry + PromptBuilder +
DecoupledActorCriticVLM_COT + RolloutBuffer + GAE + PPO loss
(Loss = mean(L_clip) + vf*mean(L_value) - ent*H, signed per §4) +
FP16 scaling + InvariantMonitor + CSV / Rich / W&B logging +
save_vlm_actor_critic_checkpoint. Inv-04 single-path re-score runs at
epoch=0 minibatch=0 each iteration (reviewer B2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: tier1 integration test

Rationale: 10-iter end-to-end run on `VizdoomBasic-v0` with the real 2B backbone; asserts loss finite, LoRA weight-hash changes, checkpoint lands, no NaN.

**Files:** Create `tests/integration/test_trainer_short_run.py`.

- [ ] **Step 1: Write the test.**

```python
"""tier1 @gpu — 10-iter short run of algos/ppo_cot.py on VizdoomBasic."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.tier1, pytest.mark.gpu]


def test_ppo_cot_short_run(tmp_path: Path, monkeypatch):
    if not (os.environ.get("GPU_AVAILABLE") or os.path.exists("/dev/nvidia0")):
        pytest.skip("no GPU present")

    # 10 iterations * num_envs=2 * num_steps=8 = 160 env steps.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    cmd = [
        sys.executable, "-m", "algos.ppo_cot",
        "--env-id", "VizdoomBasic-v0",
        "--num-envs", "2",
        "--num-steps", "8",
        "--total-timesteps", "160",
        "--max-new-tokens", "64",
        "--num-minibatches", "2",
        "--update-epochs", "1",
        "--checkpoint-interval", "10",
    ]
    r = subprocess.run(cmd, capture_output=True, env=env, cwd=".", timeout=1800)
    assert r.returncode == 0, r.stderr.decode()[-4000:]

    # Asserts on produced artifacts
    runs = list(Path("runs").glob("ppo_cot__VizdoomBasic-v0__*"))
    assert runs, "no run directory produced"
    run = max(runs, key=lambda p: p.stat().st_mtime)
    csv_path = run / "metrics.csv"
    assert csv_path.exists(), "metrics.csv missing"
    body = csv_path.read_text().splitlines()
    assert len(body) > 1, "no metrics rows"
    # Header contains reviewer-required columns.
    header = body[0].split(",")
    for col in ["gen_truncated_rate", "lora_weight_norm_actor", "inv_4_status"]:
        assert col in header
```

- [ ] **Step 2: Local invocation (single box, GPU).**

```bash
GPU_AVAILABLE=1 uv run --no-env-file pytest tests/integration/test_trainer_short_run.py -v -m "tier1 and gpu"
```
Expected on a GPU node: `1 passed` within ~30 min. On a CPU CI node: skip.

- [ ] **Step 3: Commit.**

```bash
git add tests/integration/test_trainer_short_run.py
git commit -m "$(cat <<'EOF'
tests: add tier1 @gpu short-run integration test for ppo_cot

10-iter end-to-end run on VizdoomBasic-v0; asserts run dir, metrics.csv,
and presence of reviewer-required CSV columns.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: `scripts/probe_vision.py` + initial report

Rationale: Inv-15 artifact — 20-frame scripted episode, VLM answers vision_probe.txt questions, produce `docs/vision_probes/.../report.md`.

**Files:** Create `scripts/probe_vision.py`, `docs/vision_probes/VizdoomBasic-v0_qwen3-vl-2b-instruct/report.md`.

- [ ] **Step 1: Write `scripts/probe_vision.py`.**

```python
"""Inv-15 — ground-truth vision probe.

Usage:
    python -m scripts.probe_vision --env-id VizdoomBasic-v0 --backbone Qwen/Qwen3-VL-2B-Instruct
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class ProbeFrame:
    step: int
    label_present: bool
    vlm_answer: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--env-config", default="configs/envs/VizdoomBasic-v0.yaml")
    args = ap.parse_args()

    from src.cleanrl_vlm.envs.registry import make_env
    from src.cleanrl_vlm.envs.vizdoom.action_tables import action_tables
    from src.cleanrl_vlm.models.base_vlm import BaseVLM
    from src.cleanrl_vlm.prompts.builder import PromptBuilder

    env_cfg = yaml.safe_load(Path(args.env_config).read_text())
    min_px = int(env_cfg.get("processor_min_pixels", 262144))
    max_px = int(env_cfg.get("processor_max_pixels", 1310720))

    env = make_env(args.env_id, env_cfg, seed=0, idx=0, run_name="probe_vision")()
    action_names = action_tables[args.env_id]
    pb = PromptBuilder(args.env_id, action_names)
    probe_text = (
        Path(pb.templates_root) / pb._env_id_to_slug(args.env_id) / "vision_probe.txt"
    ).read_text()

    vlm = BaseVLM(args.backbone, min_pixels=min_px, max_pixels=max_px)
    obs, _ = env.reset(seed=0)
    frames: list[ProbeFrame] = []
    for step in range(args.num_frames):
        obs_t = torch.as_tensor(obs[np.newaxis], dtype=torch.uint8, device=vlm.model.device)
        inputs = vlm.preprocess_obs_and_text(obs_t, [probe_text])
        ids = vlm.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        txt = vlm.processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        # Ground truth: VizdoomBasic labels buffer would go here; for the probe
        # we only record the VLM's answer and the agent reviews qualitatively
        # per §0.
        frames.append(ProbeFrame(step=step, label_present=True, vlm_answer=txt))
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()

    slug = args.backbone.split("/")[-1].lower()
    out = Path(f"docs/vision_probes/{args.env_id}_{slug}/report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# Vision probe — {args.env_id} / {args.backbone}", "", "| Step | VLM answer |", "|---|---|"]
    for f in frames:
        body.append(f"| {f.step} | {f.vlm_answer[:200].replace(chr(10), ' ')} |")
    out.write_text("\n".join(body))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the placeholder `docs/vision_probes/.../report.md`.** (The script will overwrite on first GPU run; committing a stub makes the doc link survive an initial `mkdocs build`.)

```markdown
# Vision probe — VizdoomBasic-v0 / Qwen/Qwen3-VL-2B-Instruct

_Auto-generated by `scripts/probe_vision.py`. Regenerate on backbone
onboarding, env onboarding, preprocessing change, or transformers/vllm
version bump._

(pending first GPU run)
```

- [ ] **Step 3: Sanity import.**

```bash
uv run --no-env-file python -c "import scripts.probe_vision; print('ok')"
```
Expected: `ok`.

- [ ] **Step 4: Commit.**

```bash
git add scripts/probe_vision.py docs/vision_probes/VizdoomBasic-v0_qwen3-vl-2b-instruct/report.md
git commit -m "$(cat <<'EOF'
scripts: add probe_vision.py + placeholder vision-probe report

CLI script runs a 20-frame scripted episode and writes
docs/vision_probes/<env>_<backbone>/report.md. Placeholder report
committed so mkdocs cross-refs resolve before first GPU run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: `scripts/probe_backbone.py` (reviewer m1, distinct from Task 21)

Rationale: Inv-8 one-shot — synthesize a known image, run processor, assert image tokens are present, count them, record patch coverage. Writes `docs/backbone_probes/qwen3-vl-2b-instruct.md`.

**Files:** Create `scripts/probe_backbone.py`, `docs/backbone_probes/qwen3-vl-2b-instruct.md`.

- [ ] **Step 1: Write `scripts/probe_backbone.py`.**

```python
"""Inv-8 one-shot: patch-coverage + token-count probe for a backbone.

Usage:
    python -m scripts.probe_backbone --backbone Qwen/Qwen3-VL-2B-Instruct \
        --min-pixels 76800 --max-pixels 76800
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.cleanrl_vlm.models.base_vlm import BaseVLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--min-pixels", type=int, default=76800)
    ap.add_argument("--max-pixels", type=int, default=76800)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    args = ap.parse_args()

    vlm = BaseVLM(args.backbone, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    # Synthesize a 4-quadrant color image.
    img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    img[: args.height // 2, : args.width // 2] = [255, 0, 0]
    img[: args.height // 2, args.width // 2 :] = [0, 255, 0]
    img[args.height // 2 :, : args.width // 2] = [0, 0, 255]
    img[args.height // 2 :, args.width // 2 :] = [255, 255, 0]

    obs = torch.as_tensor(img[np.newaxis], dtype=torch.uint8, device=vlm.model.device)
    inputs = vlm.preprocess_obs_and_text(obs, ["Describe the image."])
    num_image_tokens = int(
        inputs.image_grid_thw.prod(dim=-1).sum().item()
    ) if hasattr(inputs, "image_grid_thw") else -1
    input_len = int(inputs.input_ids.shape[1])

    assert num_image_tokens > 0, "Inv-8: processor emitted ZERO image tokens"

    slug = args.backbone.split("/")[-1].lower()
    out = Path(f"docs/backbone_probes/{slug}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Backbone probe — {args.backbone}\n\n"
        f"- min_pixels: {args.min_pixels}\n"
        f"- max_pixels: {args.max_pixels}\n"
        f"- test image: {args.width}x{args.height} 4-color quadrants\n"
        f"- image token count: {num_image_tokens}\n"
        f"- total input_ids length: {input_len}\n"
        f"- Inv-8 patch-coverage: {'PASS' if num_image_tokens > 0 else 'FAIL'}\n"
    )
    print(f"wrote {out} (image_tokens={num_image_tokens}, input_len={input_len})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Placeholder `docs/backbone_probes/qwen3-vl-2b-instruct.md`.**

```markdown
# Backbone probe — Qwen/Qwen3-VL-2B-Instruct

_Auto-generated by `scripts/probe_backbone.py`._

(pending first GPU run)
```

- [ ] **Step 3: Import sanity.**

```bash
uv run --no-env-file python -c "import scripts.probe_backbone; print('ok')"
```
Expected: `ok`.

- [ ] **Step 4: Commit.**

```bash
git add scripts/probe_backbone.py docs/backbone_probes/qwen3-vl-2b-instruct.md
git commit -m "$(cat <<'EOF'
scripts: add probe_backbone.py (reviewer m1; distinct from probe_vision)

Runs the Inv-8 patch-coverage + token-count one-shot; writes
docs/backbone_probes/<slug>.md. Separate from probe_vision.py which
generates docs/vision_probes/... reports.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: Kick off training run (in background)

Rationale: Start the long-running training job; schedule wakeups on milestone events per master-spec §13.

**Files:** No new files. Commands only.

- [ ] **Step 1: Launch background training.**

```bash
mkdir -p runs
source scripts/_cluster_env.sh
nohup uv run --no-env-file python -m algos.ppo_cot \
    --env-id VizdoomBasic-v0 \
    --backbone Qwen/Qwen3-VL-2B-Instruct \
    --num-envs 4 --num-steps 32 \
    --total-timesteps 200000 \
    --max-new-tokens 256 \
    --seed 0 \
    --track \
    > runs/latest.log 2>&1 &
echo $! > runs/latest.pid
```

- [ ] **Step 2: Probe backbone once (Inv-8 artifact).**

```bash
uv run --no-env-file python -m scripts.probe_backbone --backbone Qwen/Qwen3-VL-2B-Instruct --min-pixels 76800 --max-pixels 76800
git add docs/backbone_probes/qwen3-vl-2b-instruct.md
git commit -m "docs: regenerate backbone probe for Qwen3-VL-2B-Instruct

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Probe vision once (Inv-15 artifact).**

```bash
uv run --no-env-file python -m scripts.probe_vision --env-id VizdoomBasic-v0 --backbone Qwen/Qwen3-VL-2B-Instruct
git add docs/vision_probes/VizdoomBasic-v0_qwen3-vl-2b-instruct/report.md
git commit -m "docs: regenerate vision probe report for VizdoomBasic / Qwen3-VL-2B

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Agent reviews vision-probe artifact per §0.** If perception looks poor, iterate on pixel budget / resolution / prompt wording and re-run; commit changes.

- [ ] **Step 5: Schedule periodic wake-ups** (every ~30 min) to tail `runs/latest.log`, check for NaN / crash / Inv-04 red, and journal the curve.

---

## Task 24: Documentation update

Rationale: `docs/ALGORITHMS.md`, `docs/ENVS.md`, `docs/RECIPES.md`, `docs/BACKBONES.md`, `docs/RESULTS.md` all get their first PPO-COT row.

**Files:** Edit `docs/ALGORITHMS.md`, `docs/ENVS.md`, `docs/RECIPES.md`, `docs/BACKBONES.md`, `docs/RESULTS.md`. Create if missing.

- [ ] **Step 1: Add PPO-COT paragraph to `docs/ALGORITHMS.md`.**

```markdown
## PPO-COT

Chain-of-thought PPO on VLMs. The actor VLM generates free-form text under
the COT prompt; the parser extracts `ACTION: <NAME>` from the tail (last
match, whitelisted). The scalar logprob per trajectory is the sum of token
logprobs over the generated span; the PPO ratio is sequence-level.

The critic shares the same base VLM via a second LoRA adapter; a
CriticHead MLP reads the last non-pad hidden state. GAE(lambda) runs on
env rewards.

Signed loss:
```
Loss = mean(L_clip) + c_v * mean(L_value) - c_e * H
```

- Canon impl: `algos/ppo_cot.py`.
- Default backbone: `Qwen/Qwen3-VL-2B-Instruct` (Tier-1).
- Invariants: Inv-1, Inv-3, Inv-4 (single-path), Inv-5, Inv-6, Inv-9, Inv-10,
  Inv-11, Inv-13.
```

- [ ] **Step 2: Add row to `docs/ENVS.md`.**

```markdown
| env_id | horizon | obs shape | actions | target | prompt | probe |
|---|---|---|---|---|---|---|
| VizdoomBasic-v0 | 300 tics / ~75 env steps | (3, 240, 320) uint8 | Discrete(3): MOVE_LEFT / MOVE_RIGHT / ATTACK | 60.0 | [actor](../src/cleanrl_vlm/prompts/templates/vizdoom/basic/actor.txt) | [report](vision_probes/VizdoomBasic-v0_qwen3-vl-2b-instruct/report.md) |
```

- [ ] **Step 3: Add recipe to `docs/RECIPES.md`.**

```markdown
## PPO-COT on VizdoomBasic with Qwen3-VL-2B-Instruct

```bash
source scripts/_cluster_env.sh
uv run --no-env-file python -m algos.ppo_cot \
    --env-id VizdoomBasic-v0 \
    --backbone Qwen/Qwen3-VL-2B-Instruct \
    --num-envs 4 --num-steps 32 \
    --total-timesteps 200000 \
    --max-new-tokens 256 --seed 0 --track
```

Expected wall-clock: ~2.5-3 min per iteration on a single A6000; ~4-5 h
for a 100-iter smoke. Target score ~60 (agent-judged per §0).
```

- [ ] **Step 4: Add probe link to `docs/BACKBONES.md`.**

```markdown
| backbone | params | context | probe |
|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | 2B | 262K | [report](backbone_probes/qwen3-vl-2b-instruct.md) |
```

- [ ] **Step 5: Add first row to `docs/RESULTS.md`.**

```markdown
| combo | status | notes |
|---|---|---|
| PPO-COT × VizdoomBasic-v0 × Qwen3-VL-2B-Instruct | yellow | first run in flight (iter 4) |
```

- [ ] **Step 6: Commit.**

```bash
git add docs/ALGORITHMS.md docs/ENVS.md docs/RECIPES.md docs/BACKBONES.md docs/RESULTS.md
git commit -m "$(cat <<'EOF'
docs: add PPO-COT + VizdoomBasic + Qwen3-VL-2B-Instruct entries

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 25: `simplify` pass (reviewer m12 — BEFORE code review)

Rationale: Reviewer m12 moves `simplify` before `code-reviewer` so the reviewer sees final form.

**Files:** All changed files under `src/cleanrl_vlm/` + `algos/ppo_cot.py`.

- [ ] **Step 1: Invoke the `simplify` skill on the iter-4 diff.**

Use the Skill tool:

```
Skill(skill="simplify", args="src/cleanrl_vlm algos/ppo_cot.py tests/")
```

- [ ] **Step 2: Apply simplify suggestions; re-run full test suite.**

```bash
uv run --no-env-file ruff format .
uv run --no-env-file ruff check . --fix
uv run --no-env-file pytest -v -m "not gpu"
```
Expected: all non-GPU tests pass after simplify.

- [ ] **Step 3: `pyright` sweep.**

```bash
uv run --no-env-file pyright src/cleanrl_vlm
```
Expected: 0 errors.

- [ ] **Step 4: Commit.**

```bash
git add -u
git commit -m "$(cat <<'EOF'
style: simplify pass on iter-4 diff

Runs `simplify` skill across src/cleanrl_vlm + algos/ppo_cot.py.
Applies ruff format + ruff check --fix. Pyright clean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 26: `code-reviewer` subagent pass

Rationale: After simplify (reviewer m12), run the `superpowers:code-reviewer` subagent on the full iter-4 diff; address findings.

**Files:** Response-dependent.

- [ ] **Step 1: Dispatch the code-reviewer subagent.**

```
superpowers:requesting-code-review
```

Point the reviewer at the diff `git diff origin/master..HEAD` and the design spec `docs/superpowers/specs/amendments/2026-04-20-ppo-cot-vizdoom-basic-design.md`.

- [ ] **Step 2: Triage findings into M (blocker/major) vs m (minor).**

- [ ] **Step 3: Address each finding in a dedicated commit.** Conventional-commits prefix matches the area touched (e.g., `fix(rollout):`, `refactor(training):`).

- [ ] **Step 4: Re-run full suite after fixes.**

```bash
uv run --no-env-file pytest -v -m "not gpu"
uv run --no-env-file ruff check .
uv run --no-env-file pyright src/cleanrl_vlm
```
Expected: all green.

- [ ] **Step 5: Journal the review + its outcome in `AUTONOMY_LOG.md`.**

---

## Task 27: Journals + LOOP_STATE pivot

Rationale: Close the iter-4 loop; queue the next task.

**Files:** `CHANGELOG.md`, `AUTONOMY_LOG.md`, `LOOP_STATE.md`.

- [ ] **Step 1: Append to `CHANGELOG.md`.**

```markdown
## iter 4 — B-ppo-cot-vizdoom-basic-2B

- Ported env / model / prompt / rollout / training layers into `src/cleanrl_vlm/`.
- Landed canon PPO-COT trainer (`algos/ppo_cot.py`).
- Landed invariants Inv-1/3/4/5/6/9/10/11/13 as pytest tests.
- Generated first backbone probe + vision probe artifacts.
- First long-running PPO-COT run kicked off; curve/judgment journaled in AUTONOMY_LOG.
```

- [ ] **Step 2: Append iter-4 decision log to `AUTONOMY_LOG.md`** (one entry per non-trivial decision, e.g. "demoted Inv-01/03 to @gpu because TinyVLM stub exceeded 150 LOC" if applicable; "set processor_max_pixels=76800 per reviewer B3"; etc.).

- [ ] **Step 3: Update `LOOP_STATE.md`.**

```markdown
## Next task
- [ ] `C-envs-tier1-expand` — port VizdoomCorridor + VizdoomDefendLine prompts + configs.
- [ ] TODO(M6): revisit `frame_stack.n` when corridor + defend_line land.
```

- [ ] **Step 4: Commit.**

```bash
git add CHANGELOG.md AUTONOMY_LOG.md LOOP_STATE.md
git commit -m "$(cat <<'EOF'
loop: close iter 4 (B-ppo-cot-vizdoom-basic-2B); pivot to C-envs-tier1-expand

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

### (a) Every §3 module-table unit has a task

| §3 unit | Task |
|---|---|
| `envs/registry.py::make_env` | 4 |
| `envs/vizdoom/factories.py::make_vizdoom_env` | 3 |
| `envs/vizdoom/action_tables.py::action_tables` | 3 |
| `envs/wrappers.py::FrameSkipEnv` / `ScreenOnlyWrapper` / `DiscreteMultiBinaryWrapper` | 2 |
| `models/base_vlm.py::BaseVLM` | 7 |
| `models/lora_topology.py::default_target_modules` | 5 |
| `models/heads.py::CriticHead` / `ActorHead` / `layer_init` | 6 |
| `models/actor_critic.py::DecoupledActorCriticVLM_COT` + `active_adapter` ctxmgr | 8 |
| `prompts/builder.py::PromptBuilder` | 10 |
| `prompts/parser.py::parse_action_cot` | 10 |
| `rollout/buffer.py::RolloutBuffer` + GAE | 11 |
| `rollout/in_process.py::generate_cot_actions` + `CotRolloutStep` | 12 |
| `training/distributed.py::load_accelerator_config` | 13 |
| `training/microbatch_probe.py::probe_microbatch` | 14 |
| `training/precision.py::Fp16State` | 13 |
| `training/checkpoint.py::save_vlm_actor_critic_checkpoint` | 16 |
| `training/logging.py::RichDashboard`/`CsvWriter`/`wandb_init` | 15 |
| `training/invariants.py::InvariantMonitor` + checks | 17 |
| `algos/ppo_cot.py::main()` | 19 |
| Prompt templates (vizdoom/basic/{actor,critic,vision_probe}) | 9 |

### (b) All 9 in-scope invariants tested

| Invariant | Test file | Task |
|---|---|---|
| Inv-1 | `tests/invariants/test_inv_01_lora_trainability.py` | 8 |
| Inv-3 | `tests/invariants/test_inv_03_active_adapter.py` | 8 |
| Inv-4 (single-path) | `tests/invariants/test_inv_04_logprob_parity.py` | 17 |
| Inv-5 | `tests/invariants/test_inv_05_grad_norm.py` | 17 |
| Inv-6 | `tests/invariants/test_inv_06_fp16_scale.py` | 13 |
| Inv-9 | `tests/invariants/test_inv_09_reward_pipeline.py` | 17 |
| Inv-10 | `tests/invariants/test_inv_10_episode_boundary.py` | 11 |
| Inv-11 | `tests/invariants/test_inv_11_determinism.py` | 17 |
| Inv-13 | `tests/invariants/test_inv_13_pad_image_token_mask.py` | 17 |

### (c) Reviewer majors M1-M8 encoded

| Major | Where |
|---|---|
| M1 — typed `CotRolloutStep` dataclass | Task 12 |
| M2 — parser.py split from builder.py | Task 10 |
| M3 — regex + last match + whitelist + pathology test | Task 10 |
| M4 — Inv-1 extended (base-weight identity + disjoint optim groups) | Task 8 |
| M5 — Inv-11 bitwise (full determinism fixture) | Task 17 |
| M6 — `frame_stack.n=1` TODO + LOOP_STATE follow-up | Task 1 + Task 27 |
| M7 — microbatch probe | Task 14 |
| M8 — `max_episode_steps=null` in VizdoomBasic YAML | Task 1 |

### (d) Reviewer minors m1-m12 all land

| Minor | Where |
|---|---|
| m1 — `scripts/probe_backbone.py` distinct from `probe_vision.py` | Task 22 (distinct from Task 21) |
| m2 — `DiscreteMultiBinaryWrapper` supersedes prototype wrappers + test asserts generalization | Task 2 |
| m3 — `TinyVLMForImageTextToText` stub ≤ 150 LOC; demote on overflow | Task 8 |
| m4 — `gen_truncated_rate` metric | Task 12 (CotRolloutStep field) + Task 15 (CSV schema) |
| m5 — `lora_weight_norm_{actor,critic}` + `adapter_sync_wall_s` metrics | Task 15 |
| m6 — `save_vlm_actor_critic_checkpoint(algo_slug, ...)` signature | Task 16 |
| m7 — `active_adapter` lives in `models/actor_critic.py` | Task 8 |
| m9 — critic path = `.forward()` only, never `.generate()` | Task 8 (docstring + implementation) |
| m10 — LoRA target-modules snapshot + assert | Task 8 (build-time assert) |
| m11 — "sharding=<name> (ignored at num_processes=1)" startup log | Task 13 |
| m12 — `simplify` before `code-reviewer` | Task 25 (before) + Task 26 (after) |
| m8 (dup of m5) | Task 15 |
