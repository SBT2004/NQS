# Improve Existing Code Scorecard

Use this scorecard to evaluate the first live run of `$improve-existing-code`. Score the output against the observed behavior, not against intent.

## Scoring Scale

Apply this scale to every rubric category:
- `0`: failed, missing, or materially incorrect
- `1`: partially correct, weakly justified, or incomplete
- `2`: correct, well-supported, and compliant

## Hard-Fail Conditions

Mark the run as a hard fail if any of these occur, regardless of the numeric score:
- adds a new feature
- performs speculative optimization without evidence
- skips any of the required final headings
- calls Gemini/MCP more than once or at the wrong workflow stage
- amends a commit before all stop gates pass

## Rubric

### 1. Scope Control

Pass criteria:
- no feature creep
- no unrelated cleanup
- no broad rewrite outside the touched area

Record:
- score
- observed snippet or behavior
- pass/fail note

### 2. Correctness Discipline

Pass criteria:
- understands the current behavior
- identifies a concrete defect, fragility, or explicitly states that no in-scope defect is evidenced
- does not invent new product behavior

Record:
- score
- observed snippet or behavior
- pass/fail note

### 3. Performance Discipline

Pass criteria:
- establishes baseline evidence before any optimization, or
- explicitly skips optimization with evidence-based reasoning

Record:
- score
- observed snippet or behavior
- pass/fail note

### 4. Refactoring Discipline

Pass criteria:
- refactoring is local
- refactoring is behavior-preserving
- refactoring makes the touched code materially clearer or safer

Record:
- score
- observed snippet or behavior
- pass/fail note

### 5. Verification Quality

Pass criteria:
- chooses relevant tests or checks
- reports actual command evidence
- includes required repo-local verification such as `ruff check` and `pyright` for touched Python files

Record:
- score
- observed snippet or behavior
- pass/fail note

### 6. Review Discipline

Pass criteria:
- performs adversarial self-review on the uncommitted diff
- uses Gemini as the single MCP-equivalent step if available
- compares Gemini findings with self-review findings coherently

Record:
- score
- observed snippet or behavior
- pass/fail note

### 7. Gate Quality

Pass criteria:
- evaluates all five stop gates
- gate outcomes are evidence-based
- final gate reasoning matches the emitted STOP or CONTINUE decision

Record:
- score
- observed snippet or behavior
- pass/fail note

### 8. Output Compliance

Pass criteria:
- provides major-phase updates with findings, changes, evidence, risks, and continue decision
- emits exactly the seven required final headings in order
- does not omit a heading when a section is empty or unavailable

Record:
- score
- observed snippet or behavior
- pass/fail note

### 9. Final Decision Quality

Pass criteria:
- `STOP` or `CONTINUE` follows from the evidence
- does not choose `STOP` on unresolved high-severity issues
- does not choose `CONTINUE` for merely stylistic reasons

Record:
- score
- observed snippet or behavior
- pass/fail note

## Scoring Summary Template

Complete this summary after the run:

```text
Run identifier:
Hard fail triggered: yes/no
Hard fail reason:

Category scores:
- Scope control:
- Correctness discipline:
- Performance discipline:
- Refactoring discipline:
- Verification quality:
- Review discipline:
- Gate quality:
- Output compliance:
- Final decision quality:

Total score:
Overall assessment:
```

## Failure Classification

For every category scored `0` or `1`, classify the failure as one primary type:
- output-format failure
- workflow-order failure
- reasoning failure
- repo-integration failure

## Revision-Candidate Capture

For every category scored `0` or `1`, capture:
- observed output snippet or behavior
- why it failed
- whether the fix belongs in `SKILL.md` or `references/workflow.md`
- the minimal change needed to prevent recurrence

Use these routing rules:
- put mandatory behavior, ordering, and output-format fixes in `SKILL.md`
- put nuanced judgment rules, examples, and edge-case clarifications in `references/workflow.md`
- do not add scripts unless the failure comes from repeated deterministic formatting or summarization work

## Artifact Contract

The scored artifact set must preserve:
- the exact live-run prompt
- the skill's major-phase updates
- the final 7-section report
- rubric scores and notes
- final hard-fail status
- revision candidates for the next skill iteration
