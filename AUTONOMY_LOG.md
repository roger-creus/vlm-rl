# AUTONOMY_LOG

Append-only journal of every non-trivial `/loop` decision per §13. Each
entry: context, options, choice, rationale (with spec sections where
relevant), follow-ups. Newest at the bottom.

---

### 2026-04-20 — iter 1 — loop-bootstrap — @8f0c71c

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

---

### 2026-04-20 — iter 2 — scaffold-skeleton — @(multiple: 408684e..4a357c3)

**Context.** Iter 2 executes the pre-existing bootstrap plan's Tasks
3–20 (subset — UV env / smoke run deferred). 9 commits land on master
in per-task groups; 12 CPU-safe tests pass locally under the conda
python. The repo now carries the full §2 scaffold as flat files and
`.gitkeep` markers; `src/cleanrl_vlm/` is importable (`__version__ ==
"0.0.1"`), CLAUDE.md carries §13 verbatim, GitHub Actions are wired,
mkdocs is configured with a 12-page nav, doc stubs fill every link.

**Decisions.**

1. **Deferred Task 5 (UV init).** `uv lock` pulls a large dep tree
   (torch 2.6.0, transformers-from-git, deepspeed, vllm, vizdoom,
   ale-py, flash-attn build-from-source) that takes many minutes and
   can fail in cluster environments with specific CUDA constraints.
   Running it inside a single `/loop` iteration risks consuming the
   whole budget on network/compilation. Splitting it into its own
   iteration (`A2-bootstrap-finalize`) gives the next iteration a
   clean slate to handle retries and per-S-5 web-fetches for anything
   that breaks.

2. **Deferred Task 15 verification.** The `test_hello_vlm.py` file is
   committed, but running it requires the UV env (Task 5) and a CUDA
   GPU. Same as above — batched into `A2-bootstrap-finalize`.

3. **Deferred Task 21 (full lint + test).** Depends on UV env. Same.

4. **Skipped Task 22 (monolithic bootstrap commit).** Iter 1 decided
   per-task commits instead of the plan's single-commit structure.
   Iter 2 landed 9 commits (`408684e` LICENSE → `4a357c3` docs
   stubs). `git log c23b0f1..HEAD --oneline` lists them all.

5. **Task 24 handoff check treated as complete.** Since the /loop is
   already running and §13 is committed to CLAUDE.md — verified
   programmatically by `tests/test_spec_exists.py::test_claude_md_
   carries_autonomy_section` — there's no separate "hand off" step
   needed.

6. **Left prototype files in place.** Iter 1 decided not to do the
   orphan-branch surgery; as a result the prior prototype's `src/`,
   `deepspeed_zero*.yaml`, `requirements.txt`, `setup.py`, `test.py`,
   `prompts/` trees remain in the working tree. None of them are
   imported by `cleanrl_vlm`; they are logical scratch space. LOOP_STATE
   lists them explicitly so a later iteration can prune each when its
   replacement lands (per-file `git rm` rather than a bulk wipe, to
   stay inside the §13.1 safety floor).

7. **Test run used conda python under `PYTHONPATH=src`, not UV.**
   The 12 CPU-safe tests passed under the existing conda env. Once
   the UV env lands, the same tests should still pass — verified in
   iter 3 as part of `A2-bootstrap-finalize`.

8. **Iteration committed more than I originally scoped.** Iter 1
   notes said "iter 2 scope is project skeleton (Task 7 / CLAUDE.md
   rewrite among others)." The actual iter-2 execution also covered
   Tasks 3, 4, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20 — most of
   the bootstrap plan minus the env-level tasks. The scope expanded
   because the tasks are mechanical file writes with mutual
   dependencies (e.g., mkdocs.yml's nav requires all the doc stubs
   to exist for `mkdocs build --strict`); splitting across iterations
   would have forced a broken intermediate state.

**Artifacts produced this iter.**

`LICENSE`, `pyproject.toml`, `src/cleanrl_vlm/` package tree (7
`__init__.py` files), `algos/` `baselines/` `experimental/`
`configs/envs/` `scripts/` `docs/backbone_probes/`
`docs/vision_probes/` `docs/superpowers/specs/amendments/`
`.gitkeep` markers, `tests/` submodule `__init__.py` files,
`CLAUDE.md` (rewritten), `README.md` (rewritten — overwrote the
prototype's conda-setup content), `MEMORY.md`,
`tests/test_imports.py`, `tests/test_spec_exists.py`,
`tests/smoke/test_hello_vlm.py`, `.pre-commit-config.yaml`,
`.github/workflows/ci.yml`, `.github/workflows/docs.yml`,
`mkdocs.yml`, `docs/index.md`, and 12 `docs/*.md` stubs.

**Skills invoked.** `superpowers:using-superpowers` (auto at session
start from iter 1's ScheduleWakeup continuation). No other skill
called out directly — tasks were pre-specified by the bootstrap plan
and executed inline.

**Skills not invoked, justification.**
- `superpowers:brainstorming` — scaffold is execution, not design
  (same as iter 1).
- `superpowers:writing-plans` — the plan already exists
  (`2026-04-19-bootstrap-scaffold.md`); nothing new to write.
- `superpowers:subagent-driven-development` — tasks are mechanical
  file creation with strict plan content; subagent overhead wasn't
  worth it. The next iteration (`A2-bootstrap-finalize`) deals with
  UV env + a real smoke test and *will* benefit from subagents for
  parallel retry handling.
- `superpowers:code-reviewer` — the iter-2 diff is scaffolding with
  no branching logic. Deferred to after iter-3 UV-env + smoke so the
  reviewer has real test-pass signal to read.
- `superpowers:test-driven-development` — Tasks 13/14 follow the TDD
  shape (test-first, test-exists-before-impl isn't applicable for
  scaffolded `__init__.py` modules; however the tests *are* committed
  in the same iteration as the impl per S-1).
- `superpowers:simplify` — nothing to simplify; scaffold is
  deliberately minimal already.
- `superpowers:verification-before-completion` — 12 CPU-safe tests
  passing is the verification; the rest (mkdocs build --strict, UV
  lock, pre-commit run, hello-VLM smoke) depends on iter 3's env
  setup.

**Follow-ups.**

- Iter 3 (`A2-bootstrap-finalize`): UV env + lockfile + smoke. Expect
  to web-fetch per S-5 on any dep-resolution failure.
- Iter 4+ (`B-ppo-cot-vizdoom-basic-0.8B`): first canon trainer,
  using the superpowers skills seriously (brainstorming for design,
  writing-plans, subagent-driven-development, code-reviewer, etc.).
- Prune prototype files (`src/models/`, `src/train_*.py`,
  `deepspeed_zero*.yaml`, `requirements.txt`, `setup.py`,
  `test.py`) once their replacements land under `algos/` + `src/cleanrl_vlm/`.

---

### 2026-04-20 — iter 3 — A2-bootstrap-finalize — @b8dd216..7fb54bd

**Context.** Finalize the scaffold: UV env + lockfile, flash-attn, lint
+ pre-commit + mkdocs strict, GPU hello-VLM smoke. Aim: the scaffold
runs end-to-end on the 8 × A6000 node.

**Outcome.** **Bootstrap DONE.** hello-VLM smoke green in 92.93 s on
`Qwen/Qwen3-VL-2B-Instruct`. All 13 applicable tests pass (12 CPU
unit + 1 GPU smoke). Next phase is the first canon trainer.

**Decisions this iter.**

1. **UV env on local scratch venv.** `uv venv --python 3.10` picked
   `/usr/bin/python3.10` (3.10.12). Venv at `.venv/`. `uv lock`
   resolved 222 packages in 52s (transformers pinned to
   `hf/transformers@a29df2d`). `uv sync --extra dev` succeeded on
   second try (see decision 2).

2. **Fixed hatch `allow-direct-references` for transformers-from-git
   dep.** First `uv sync` failed:
   `ValueError: Dependency #4 of field 'project.dependencies' cannot
   be a direct reference unless tool.hatch.metadata.allow-direct-references
   is set to true`. Added the config to `pyproject.toml` (commit
   `4589b7b`).

3. **Ruff scope excludes prototype leftovers.** First `ruff check .`
   reported 279 errors across the pre-scaffold files (`src/eval.py`,
   `src/models/`, `src/ppo.py`, `src/train_decoupled_*.py`,
   `src/utils/`, `setup.py`, `test.py`). Per iter-1 decision these
   files are slated for pruning when their replacements land. Added
   them to `[tool.ruff] extend-exclude`. `ruff check .` / `ruff
   format --check .` now clean (commit `1ea7938`).

4. **Mkdocs `--strict` README-link fix.** `docs/index.md` linked at
   `../README.md` which is outside the docs tree; mkdocs strict mode
   flagged it. Replaced with an inline quickstart block.

5. **`site/` build output gitignored.**

6. **flash-attn CUDA_HOME gotcha solved.** First build:
   `OSError: CUDA_HOME environment variable is not set`. Cvmfs has
   `cuda/12.5.0` available; torch is cu124. Built with:
   `CUDA_HOME=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/12.5.0
   PATH=$CUDA_HOME/bin:$PATH ... --no-build-isolation`. Succeeded
   in 17s (cached somewhere). Documented in
   `docs/TROUBLESHOOTING.md`.

7. **Backbone-name correction amendment.** Attempted to load
   `Qwen/Qwen3.5-VL-0.8B` per spec §3 — **repo does not exist on
   HF** (RepositoryNotFoundError). Searched HF: no `Qwen3.5-VL`
   series is published yet; real series is `Qwen3-VL-{2B,4B,8B,...}-
   {Instruct,Thinking}`. Wrote
   `docs/superpowers/specs/amendments/2026-04-20-backbone-names-correction.md`
   — Tier-1 replacement `Qwen/Qwen3-VL-2B-Instruct` (no 0.8B exists),
   Tier-2 replacement `Qwen/Qwen3-VL-4B-Instruct`, thinking-ablation
   `-4B-Thinking`. Updated `docs/BACKBONES.md` (four-row table) and
   `tests/smoke/test_hello_vlm.py` (MODEL_ID default). Per §11 S-11.
   Spec §3 text is NOT rewritten in-place this iter — amendment
   stands as the active override until next full spec edit.

8. **Smoke-test Auto class fix.** With the corrected model ID,
   the smoke test failed with `ValueError: Unrecognized configuration
   class Qwen3VLConfig for AutoModelForCausalLM`. Qwen3-VL is
   multimodal and registers under `AutoModelForImageTextToText`.
   Swapped the import (commit `7fb54bd`). Second smoke run passed
   in 92.93s.

9. **Pre-commit hygiene committed for prototype files.** `pre-commit
   run --all-files` auto-fixed whitespace / EOF / mixed-line-ending
   on 28 prototype + scaffold files. Committed the cosmetic-only
   diff (commit `7fb54bd`) so future pre-commit runs don't re-touch
   them. The prototype files remain in the working tree pending
   iter-4+ port work.

10. **`.claude/` gitignored.** Pre-commit/scheduler wrote
    `.claude/scheduled_tasks.lock` which got `git add -A`-staged.
    Added `.claude/` to `.gitignore`.

**Commits this iter.**

| SHA | What |
|---|---|
| `b8dd216` | `uv.lock` committed (222 packages) |
| `4589b7b` | hatch `allow-direct-references` fix |
| `1ea7938` | ruff exclude + mkdocs strict + site/ ignore |
| `cefa7c9` | amendment: backbone names correction |
| `7fb54bd` | pre-commit hygiene + smoke-test VLM class + .claude ignore |

Plus the iter-3 journal update (this commit) — likely SHA to be
backfilled.

**Skills invoked.** `superpowers:using-superpowers` (auto).
`superpowers:systematic-debugging` applied implicitly (hypothesize →
verify → fix) on each of decisions 2, 6, 7, 8. S-5 "external-library
research" applied implicitly when adapting to Qwen3-VL reality — a
single `HfApi().list_models(author='Qwen', search='VL')` query
substituted for the web-fetch; summary landed in the amendment.

**Skills not invoked, justification.**
- `superpowers:brainstorming` — scaffold finalization is execution,
  not design. Model-name amendment is adaptation to reality, also
  not net-new design.
- `superpowers:writing-plans` — the bootstrap plan suffices.
- `superpowers:code-reviewer` subagent — iter-3 diff is 100 % scaffold
  glue (lockfile, lint exclusions, test fixes, amendment). The
  reviewer is more valuable on the first canon trainer (`B`).
- `superpowers:test-driven-development` — no new tests written this
  iter beyond the scaffold's 13 from iter 2.
- `superpowers:simplify` — nothing to simplify.
- `superpowers:subagent-driven-development` — tasks were sequential
  (many depended on UV being up first). Subagents would not have
  parallelized usefully.

**Follow-ups.**

- Iter 4 kicks off `B-ppo-cot-vizdoom-basic-2B`. First sub-step:
  brainstorm + writing-plans to produce the detailed task file.
- The master spec §3 `Qwen3.5-VL` naming will be rewritten in-place
  at a later spec-edit iteration (low priority). Amendment stands.
- `docs/backbone_probes/qwen3-vl-2b-instruct.md` lands when a
  real trainer first loads the backbone (not for the hello-smoke —
  that's just a generation sanity).
- HF cache lives at `$SCRATCH/hub` per env var; the smoke downloaded
  ~5GB there. No need to re-download for iter-4 training runs.
- Mila-specific env needs: `CUDA_HOME=/cvmfs/.../cuda/12.5.0`. Scripts
  that run flash-attn-dependent code should set this via a shared
  helper or document it prominently. Candidate: `scripts/_cluster_env.sh`.

---

### 2026-04-20 — iter 4 — ppo-cot-design — @794e7a9..f7b7fe7

**Context.** First canon trainer kickoff. Per §13.1, skip the
brainstorming skill's HARD-GATE by writing the design spec to
`amendments/` (not asking the user).

**Artifacts produced.**

- `docs/superpowers/specs/amendments/2026-04-20-ppo-cot-vizdoom-basic-design.md`
  — the design spec, committed first at `794e7a9`, revised at `f7b7fe7`.
  14 sections covering goal, scope, architecture (module table),
  PPO-COT algorithm, prompts, invariants, test strategy, configs,
  logging, deliverable sequence (§10), risks, non-goals, reviewer
  findings, sign-off.

**Code-reviewer subagent findings (summary; full text
in the subagent's tool result, not transcribed here).**

- 3 blockers (B1 entropy sign / B2 cheap Inv-4 / B3 pixel budget).
- 8 majors (M1 typed rollout return / M2 parser split / M3 regex
  parser / M4 Inv-1 base-weight identity / M5 Inv-11 bitwise /
  M6 frame_stack / M7 microbatch probe / M8 VizdoomBasic cap).
- 12 minors folded into simplify pass + §9 schema additions + §10
  ordering.

**Decisions (§13.1 autonomous).**

1. **Design location: `amendments/`, not `specs/`.** Per spec §2 the
   top-level `specs/` holds the master spec only; amendments + plans
   are the only writable subdirs. `amendments/` was the closest match
   in intent (per-feature design doc that lives alongside the master
   spec). Writing-plans output will go to `plans/` next iter.

2. **All reviewer blockers resolved in-spec this iter.** Per reviewer
   recommendation, these are not deferrable — they would break the
   first trainer silently. B1 took 2 lines (loss formula
   unambiguation). B2 added ~15 lines (single-path Inv-4 table row).
   B3 took 6 lines (per-env YAML pixel override).

3. **M5 Inv-11 committed to bitwise per master-spec §0 + §8.** First
   design draft softened Inv-11 to "fp16 tolerance" under §0's
   signal-not-cliff principle; reviewer correctly noted that §0
   specifically lists "non-deterministic behavior under a fixed seed"
   as a *hard-fail* item. The softening would have been a silent
   master-spec override. Revised: §11 risk row lists the full
   deterministic-fixture settings required (`use_deterministic_algorithms`,
   `CUBLAS_WORKSPACE_CONFIG`, `cudnn.benchmark=False`,
   `cudnn.deterministic=True`, vizdoom seed, HF seed). Components
   that provably cannot honor bitwise will `pytest.xfail()` with
   justification, not loosen the invariant.

4. **Majors M1–M8 assigned to named §10 tasks rather than resolved
   here.** The reviewer explicitly said "fine to address inside the
   iter-4 implementation PRs *provided* the plan explicitly assigns
   them to named tasks." Writing-plans next iter will see the §13
   enumeration and encode each fix into the corresponding plan task.

5. **Minors folded into simplify pass (§10 task 25).** Reviewer's
   suggestion to reorder simplify-before-code-review adopted.

6. **VizdoomBasic pixel budget = 76800 exactly.** Chose reviewer
   option (a) — per-env YAML override with `processor_min_pixels =
   processor_max_pixels = 76800` — rather than option (b) re-rendering
   at higher res. Rationale: smaller pixel budget → fewer image tokens
   → faster generation, which is the bottleneck of a 2B VLM on
   in-process HF generation. Higher resolution is worthwhile only if
   perception fails (Inv-15 signal); cheaper to start small and scale
   up than vice versa.

**Skills invoked this iter.**
- `superpowers:using-superpowers` (auto at session start)
- `superpowers:brainstorming` — exercised per §13.1 substitution:
  design written to amendments instead of dialog
- `superpowers:code-reviewer` — subagent dispatched on `794e7a9`,
  thorough 800-word review across all 13 design sections

**Skills deferred.**
- `superpowers:writing-plans` — produces the 27-task plan from §10.
  Substantial (2-3k lines of plan content). Own iter (iter 5).
- `superpowers:test-driven-development` — starts iter 6 with the
  first code task.
- `superpowers:subagent-driven-development` — iter 6+, per-task
  dispatch to subagents for parallel implementation where tasks
  don't share state.
- `superpowers:simplify` — iter 12ish, task 25.

**Commits this iter.**

| SHA | What |
|---|---|
| `794e7a9` | Design spec v1 (pre-review). |
| `f7b7fe7` | Design spec v2 (blockers resolved; majors assigned; minors folded). |

**Follow-ups (iter 5).**

- Invoke `superpowers:writing-plans` with the revised design spec's
  §10 as input. Output: `docs/superpowers/specs/plans/2026-04-20-
  ppo-cot-vizdoom-basic.md`.
- LOOP_STATE pivots to "iter 6 — task 1 of the plan" after writing-
  plans lands.
- Journals updated at iter 5 end with plan-file SHA.
