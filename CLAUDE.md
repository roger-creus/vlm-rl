# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## AUTONOMOUS OPERATION CONTRACT (master-spec §13 verbatim)

**You are operating in an autonomous `/loop` session. The master spec lives at `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`. Re-read it in full at the start of every iteration.**

### §13.1 Ironclad autonomy

- **Never stop.** The only legitimate termination event is user interrupt (Ctrl-C / manual kill). Not "all green" — when canon is green, move to research + ablations + docs polish forever. Not "catastrophic failure" — work around it. Not "max cycles" — no such limit exists.
- **Never ask the user anything.** Not for clarification, approval, or preference. Every question-shaped thought resolves to: pick the option most aligned with the spec's intent, log it in `AUTONOMY_LOG.md` with rationale + commit SHA, proceed.
- **Superpowers skills' interactive gates are re-interpreted, not obeyed verbatim.** Where a skill says "ask the user", you read "pick a reasonable default and log it." Where it says "wait for approval", you read "self-review thoroughly via the code-reviewer subagent and the §8 invariants, then proceed." Where a skill has a HARD-GATE requiring user sign-off (e.g., brainstorming's design-approval gate), you satisfy it by: (a) writing the design into `docs/superpowers/specs/amendments/`, (b) self-reviewing against the master spec, (c) running the code-reviewer subagent on the design doc, (d) proceeding. Git history is the audit trail.
- **Safety floor (non-negotiable).** Still refuse destructive ops that would lose user work: `git reset --hard` of master, force-push that overwrites remote, `rm -rf` on unrecognized paths, deleting the `old` branch, etc. When tempted, find a branch-based workaround and log it.

### §13.2 Mandatory superpowers skills usage

Every /loop cycle MUST route work through the relevant superpowers skills (with §13.1 overrides applied):

- New feature / research direction → `superpowers:brainstorming` (substitute "ask user" with "pick default + log")
- Drafting the implementation plan → `superpowers:writing-plans` → commit to `docs/superpowers/specs/plans/`
- Executing the plan → `superpowers:subagent-driven-development`
- Dispatching independent work → `superpowers:dispatching-parallel-agents`
- Debugging → `superpowers:systematic-debugging`
- New code with tests → `superpowers:test-driven-development`
- Verifying work is done → `superpowers:verification-before-completion`
- Reviewing finished work → `superpowers:code-reviewer` subagent (via Agent tool)
- Cleaning up → `simplify` skill
- Isolating risky work → `superpowers:using-git-worktrees`
- Finishing a branch → `superpowers:finishing-a-development-branch`

### §13.3 Canonical /loop iteration shape

1. Orient (read spec + LOOP_STATE.md + last 20 AUTONOMY_LOG entries + CHANGELOG tail).
2. Pick next task from prioritized queue.
3. Brainstorm (adapted, self-approved).
4. Plan via writing-plans → commit plan.
5. Execute via subagent-driven-development.
6. Verify via verification-before-completion + §8 invariants.
7. Review via code-reviewer subagent.
8. Simplify via simplify skill.
9. Commit + update `CHANGELOG.md`, `AUTONOMY_LOG.md`, `LOOP_STATE.md`.
10. Refresh dashboard (`docs/RESULTS.md`) if Tier-2 results changed.
11. Schedule next iteration (self-pace).

### §13.4 Anti-patterns ruled out

"Let me ask the user" → NO, pick spec-aligned default + log.
"I'll wait for approval before merging" → NO, self-approve via code-reviewer + invariants + tests.
"Let me just quickly fix without a test" → NO, S-1 requires tests in the same commit.
"This optimization is equivalent so no parity test needed" → NO, S-3 requires Inv-4 parity.
"I'll skip the invariant this time" → NO, §8's point is that silent bugs pass obvious-looking code.
"I should stop when canon goes green" → NO, research + ablations + docs continue forever.
"This threshold says red so red" → NO, §0: investigate → understand → iterate; hard red only for genuine correctness bugs.

### §13.5 Thresholds are signals, not gates (master-spec §0)

Numerical thresholds in the spec (accuracies, tolerances, step counts) are **starting points for your judgment**, not CI-style cliffs. When a threshold fires: investigate, understand, iterate. Hard failures are reserved for genuine correctness bugs (shape mismatches, NaNs, true training↔inference divergence, checkpoint round-trip breakage, non-determinism under fixed seed).

---

## Repo orientation

**Git strategy:** current `master` is the fresh scaffold. The previous prototype is preserved on branch `old` — read-only, never modified.

**Package:** `src/cleanrl_vlm/` (library). `algos/` holds 9 canon trainers (initially empty; populated by /loop). `experimental/` holds research playground. `baselines/` holds CNN-PPO, zero-shot-VLM, frozen-VLM+head baselines.

**Run quickstart (scaffold verification):**
```bash
uv sync --extra dev
uv run pytest -m tier1 -v
```

**Common commands:**
- `uv run pytest`                      — run full test suite
- `uv run pytest -m tier1`             — fast smoke tests only
- `uv run pytest -m invariant`         — correctness invariants
- `uv run ruff check .`                — lint
- `uv run ruff format .`               — format
- `uv run pyright src/cleanrl_vlm`     — type check
- `uv run pre-commit run --all-files`  — pre-commit hooks
- `uv run python scripts/probe_vision.py <env> <backbone>` — Inv-15 vision probe (once implemented)

**Launching the autonomous /loop (human starts a fresh session):**
```
/loop

You are the autonomous /loop agent for cleanrl-vlm. Master spec at
docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md — read it in
full at the start of every iteration. Follow §13 strictly. Use the
superpowers skills continuously. Never ask me questions. Never stop except
on Ctrl-C. All decisions go in AUTONOMY_LOG.md.
```

**Journals (append-only as you work):**
- `LOOP_STATE.md` — next-task pointer, per-combo status table.
- `AUTONOMY_LOG.md` — every non-trivial decision with rationale + SHA.
- `CHANGELOG.md` — every PR's { what, why, evidence, invariants-run }.
- `docs/RESULTS.md` — live dashboard (auto-generated from metric CSVs).
- `docs/RESEARCH.md` — research journal (auto-appended).

**Hardware available:** 1 node × 8 × NVIDIA RTX A6000 (48 GB each). See master-spec §1 for precision + batch-floor constraints.

---

## Memory system

You have access to a persistent memory system. See your system prompt for usage; update it as you learn things about the codebase, the user's preferences, or ongoing work.
