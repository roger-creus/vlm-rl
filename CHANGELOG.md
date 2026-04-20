# Changelog

Per §11 S-12: every `/loop` iteration that changes tracked state adds
an entry here — `{ what, why, evidence, invariants-run }`. Newest on top.
One entry per iteration, not per commit within an iteration.

---

## 2026-04-20 — iter 1 — loop bootstrap

**What.**
- Created local branch `old` pointing at pre-scaffold master HEAD `964377d`
  (§1 "Git strategy").
- Added `LOOP_STATE.md`, `AUTONOMY_LOG.md`, `CHANGELOG.md` (this file) —
  the `/loop` operating surface.
- Patched `.gitignore` with Python / UV / runs / zip ignores on top of
  the pre-existing project-specific ignores.
- Tracked `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`
  and `docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md`
  (both were untracked at session start).

**Why.**
Bootstrap the `/loop` operating surface before any scaffold code lands.
The journal / state / changelog files are required by §13.3 step 9;
`old` preservation is required by §1. The spec + pre-existing bootstrap
plan must be tracked so future iterations (which re-read both at start)
are reading from the committed copy.

**Evidence.**
N/A — policy-boilerplate commit, no runtime code paths changed. No
invariants apply yet (no model, env, or training surface exists).

**Invariants run.** None applicable.

**Skills invoked.** `superpowers:using-superpowers`, `superpowers:writing-plans`.

**Adaptations to the pre-existing bootstrap plan (full rationale in
AUTONOMY_LOG).**
- Skipped orphan-branch surgery (`git rm -rf .` + `git clean -fdx`) —
  unnecessary given `old` preserves history; too destructive for the
  §13.1 safety floor in autonomous mode.
- Committing per task group instead of the plan's single Task-22
  monolithic commit — better `/loop` observability.
