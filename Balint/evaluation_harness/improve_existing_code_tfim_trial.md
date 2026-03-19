# Improve Existing Code Live Trial: TFIM Operator Work

Use this packet to run the first live evaluation of `$improve-existing-code` against the current dirty `tfim`-related changes in this repository. Freeze this packet before the run. Do not change the prompt, target files, or scoring rules mid-run.

## Trial Goal

Evaluate whether `$improve-existing-code` can handle an existing in-flight operator change without adding features, while respecting repo-local workflow rules and producing an evidence-based final decision.

## Frozen Task Statement

Improve the existing `tfim`-related code path in `src/nqs/operator.py` and its tightly related exports/tests without adding features.

## Frozen Target Surface

Treat these as the in-scope touched files for the first live trial:
- `src/nqs/operator.py`
- `tests/test_operator.py`
- `src/nqs/__init__.py`
- `nqs/__init__.py`

Treat the rest of the repository as read-only unless the workflow reveals a directly related verification need.

## Frozen Baseline Context

Use this baseline context when issuing the live run:
- The worktree is already dirty before the skill run starts.
- Existing local edits add a `tfim` operator builder, export it from both package entry points, and add tests for term construction and connected-element behavior.
- The skill must not revert or rewrite unrelated local edits in the touched files.
- Nearby verification already exists in `tests/test_operator.py`.
- Repo-local instructions live in `AGENTS.md` and must be obeyed.

## Repo Constraints to Hand the Skill

The live-run prompt must include these constraints:
- edit only inside `Balint/`
- prefer minimal, focused edits
- use the narrowest relevant verification
- run `ruff check` on touched Python files
- run `pyright` on touched Python files or the relevant package
- use Gemini second-opinion review as the MCP step if available
- do not amend a commit unless every stop gate passes

## Frozen Live-Run Prompt

Use this prompt verbatim for the first run, adjusting only the absolute workspace path if needed:

```text
Use $improve-existing-code at C:\Users\balin\.codex\skills\improve-existing-code\SKILL.md.

Work only inside C:\Users\balin\PycharmProjects\NQS\Balint.

Task context:
- Target area: the existing tfim-related operator work in src/nqs/operator.py and its tightly related exports/tests.
- Current suspicion: this in-flight change may still have correctness gaps, weak verification, unnecessary complexity, or unjustified cleanup.
- Constraints: obey AGENTS.md, preserve existing local edits unless the workflow justifies a change, do not add features, and keep changes tightly scoped to the touched area.
- Likely verification surface: tests/test_operator.py plus any narrow import, lint, type-check, or build checks required by the touched files.

Baseline context:
- The worktree is already dirty before you begin.
- Current touched files are src/nqs/operator.py, tests/test_operator.py, src/nqs/__init__.py, and nqs/__init__.py.
- The current local diff adds a tfim operator builder, exports it from both package entry points, and adds tests covering term construction and connected elements.

Follow the full $improve-existing-code workflow:
- discover repo-specific commands dynamically
- establish a baseline before making optimization claims
- perform adversarial self-review on the uncommitted diff
- use Gemini second-opinion review as the single MCP-equivalent step if it is available
- do not amend a commit until all stop gates pass

Reporting requirements:
- provide brief major-phase updates with what you found, what you changed, evidence, risks, and whether the workflow should continue
- end with exactly these headings in this order:
  1. Summary of changes
  2. Verification results
  3. Self-review findings
  4. MCP findings
  5. Improvements implemented after MCP
  6. Stop-condition evaluation by gate
  7. Final decision: STOP or CONTINUE
```

## Run Procedure

Follow this procedure exactly:
1. Capture the frozen baseline before the run:
   - current `git status --short`
   - current `git diff` for the four in-scope files
   - current contents of `AGENTS.md`, `src/nqs/operator.py`, and `tests/test_operator.py`
2. Start one live run with the frozen prompt above.
3. Let the skill execute end-to-end without changing the task framing mid-run.
4. Preserve every major-phase update from the run.
5. Preserve the final 7-section report exactly as emitted.
6. Score the run using `evaluation_harness/improve_existing_code_scorecard.md`.
7. Record only concrete revision candidates backed by observed output or behavior.

## Artifact Capture Checklist

After the run, preserve:
- the exact prompt used
- the pre-run baseline snapshots
- the major-phase updates
- the final 7-section report
- the final `git diff`
- verification command results
- Gemini/MCP review input summary and output summary
- the completed rubric and revision-candidate notes

## Success Conditions for Preparation

Preparation is complete only when:
- the trial target is fixed
- the prompt is fixed
- the run procedure is fixed
- the artifact checklist is fixed
- the scoring rubric is fixed in the companion scorecard
