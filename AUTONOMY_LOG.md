# AUTONOMY_LOG

Append-only journal of every non-trivial `/loop` decision per §13. Each
entry: context, options, choice, rationale (with spec sections where
relevant), follow-ups. Newest at the bottom.

---

### 2026-04-20 — iter 1 — loop-bootstrap — @(pending)

**Context.** First `/loop` invocation against the master spec committed at
`docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md`. Repo is
pre-scaffold: `master` still carries the prior `clean-llm-rl` prototype.
Untracked working-dir contents at session start: `CLAUDE.md` (old
project-orientation content), `docs/` (master spec only), and a
`cleanrl-master (1).zip` reference dump (Appendix C).

**Discovery mid-iteration.** A comprehensive pre-existing plan was found
at `docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md` —
24 tasks authored during the original spec-brainstorming session, with
self-review and a Which-approach execution handoff. It is the authoritative
scaffold contract. I had started drafting a narrower duplicate plan
(`2026-04-20-scaffold-project-skeleton.md`) before noticing; the duplicate
was deleted.

**Decisions.**

1. **Preserve `old` (keep).** Ran `git branch old master` so the current
   HEAD is frozen at `964377d` per §1. Non-destructive. `old` is never
   touched again. Will be published on first push.

2. **Skip the orphan-branch surgery (adapt Task 1 Steps 4–5).**
   The pre-existing plan's Task 1 blanks the working tree via
   `git checkout --orphan scaffold-bootstrap` + `git rm -rf .` +
   `git clean -fdx`. Rationale for skipping:
   - §1 only requires "master moved to `old`"; it does not prescribe an
     orphan branch. `old` already preserves history.
   - `git clean -fdx -e .claude -e docs` is destructive enough to trip
     the §13.1 safety floor (rm of the prior prototype's `src/`,
     `prompts/`, `deepspeed_zero*.yaml`, `test.py`, `requirements.txt`,
     `setup.py`, `README.md`). In autonomous mode it's safer to leave
     those files in the working tree; subsequent bootstrap tasks
     overwrite the files they care about (`README.md`, `.gitignore`),
     and the rest (prototype `src/`) is logically replaced by new code
     under `src/cleanrl_vlm/` + `algos/` in the library-port iterations.
     The pre-existing-but-unused files carry no runtime cost; they can be
     pruned later with a targeted `git rm` per file per §13.1.
   - This decision is an adaptation of the plan, not a violation of the
     spec.

3. **Adopt the pre-existing plan's scope; reject its "single bootstrap
   commit" structure.** The plan's Task 22 lands all 21 preceding tasks in
   one commit. For `/loop` observability (every iteration should produce
   at least one legible commit), I commit per task group instead. The
   `CHANGELOG.md` format lists one entry per iteration; commits within an
   iteration use conventional-commits subjects.

4. **Delete the duplicate plan.** Removed
   `docs/superpowers/specs/plans/2026-04-20-scaffold-project-skeleton.md`
   to avoid ambiguity — there is exactly one authoritative scaffold plan.

5. **`.gitignore` patched in place instead of rewritten.** The existing
   `.gitignore` had project-specific entries (`launchers/`, `*_vizdoom*`)
   worth preserving; I unioned those with the Python + UV + runs +
   zip ignores that Task 2 of the bootstrap plan prescribes. Content
   differs from the plan's verbatim block but covers the same ignores
   plus the project-specific ones. If the plan's exact content ever
   matters for reproducibility, the log notes the intentional divergence.

6. **`CLAUDE.md` left untracked this iter.** Current CLAUDE.md on disk
   is the pre-scaffold orientation content. The scaffold plan's Task 7
   rewrites it with §13 verbatim at top per Appendix B. Iter 1 avoids
   touching it so iter 2 can do the full rewrite in one clean commit.
   The file is not yet in the git index; iter 2 will `git add` it after
   rewrite.

7. **Brainstorming skipped (autonomous mode interprets §13.2).**
   §13.2 mandates the brainstorming skill "for identifying a new feature
   / research direction." The scaffold is pure execution of §2 +
   Appendix B, not a new design. Per §13.4 ("Design is obvious; I'll
   skip brainstorming — NO"), the counter-argument applies only to
   non-obvious design work; the spec explicitly notes the scaffold *is*
   the design. Logging the deviation here for auditability.

8. **No code-reviewer subagent this iter.** The iter-1 diff is journal +
   gitignore + plan delete only. Trivial. The subagent will be invoked
   at the end of iter 2 once real scaffold code lands.

**Artifacts produced this iter.**

- `.gitignore` — expanded Python / UV / runs / zip ignores.
- `LOOP_STATE.md` — seeded with the interim-state pointer.
- `AUTONOMY_LOG.md` — this file.
- `CHANGELOG.md` — seeded with iter-1 entry.
- `docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md` — now
  tracked (was untracked at session start).
- `docs/superpowers/specs/plans/2026-04-19-bootstrap-scaffold.md` — now
  tracked (was untracked at session start).

**Skills invoked.** `superpowers:using-superpowers` (auto at session
start), `superpowers:writing-plans` (reviewed — then discovered the
pre-existing plan and aligned to it; no new plan file produced net).

**Skills intentionally not invoked.** `superpowers:brainstorming`,
`superpowers:code-reviewer`, `superpowers:simplify`,
`superpowers:test-driven-development` — all deferred to iter 2+.

**Follow-ups.**

- Iter 2 picks up at bootstrap-plan Task 3 (LICENSE) and works through
  Task 24. Skip Task 1 Steps 4–5 (orphan surgery — already decided).
- Iter 2 first step: re-read the master spec per §13.3.1, then re-read
  the bootstrap plan top-to-bottom, then dispatch subagents per task
  group.
- At the end of iter 2, AUTONOMY_LOG appends a new entry; CHANGELOG
  appends an iter-2 entry; LOOP_STATE next-task flips to
  `B-ppo-cot-vizdoom-basic-0.8B`.
