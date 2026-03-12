# Agent Instructions

## Project Context

- Treat [Balint](C:\Users\balin\PycharmProjects\NQS\Balint) as the only writable
  project workspace.
- Do not create new project files outside
  [Balint](C:\Users\balin\PycharmProjects\NQS\Balint). Future files, folders,
  tests, demos, and docs must all live inside that directory.
- Treat sibling folders such as `Anas/` as read-only unless the user explicitly
  says otherwise.

## Code Style
- Edit only files inside
  [Balint](C:\Users\balin\PycharmProjects\NQS\Balint). Treat the rest of the
  repository as read-only.
- Keep changes minimal and consistent with surrounding code.
- Prefer reusable logic in `.py` modules. Use notebooks for experiments, demos,
  plots, and exploratory validation.
- Demos must be notebook-first:
  use `demo.ipynb` for the user-facing demo and `demo_helper.py` for reusable
  support code in the same `demos/` folder.
- Do not leave important shared logic only in notebooks if it belongs in a
  module.
- Keep subsystem boundaries clean:
  graph/lattice, hilbert space, operators, models, samplers, variational state,
  driver, observables.
- Prefer pure functions where practical, especially in physics utilities and
  estimators.
- Keep array shape conventions and spin-value conventions explicit in code and
  docstrings.
- Prefer batched JAX operations over per-sample Python loops when the logic is
  part of model evaluation, sampling, or estimators.
- Keep models, samplers, and operators decoupled. Do not special-case one
  ansatz in shared infrastructure unless the behavior is intentionally
  architecture-specific.
- Preserve a notebook-friendly public API. User-facing code should continue to
  read like:
  `graph -> hilbert -> operator -> model -> sampler -> variational state -> driver -> observables`.
- RBM is the first reference ansatz. FFNN and CNN should fit the same interface
  rather than introduce parallel workflows.
- Treat von Neumann entropy as a small-system exact tool. Treat Renyi-2 via
  SWAP as the scalable NQS entanglement observable.

## Workflow
1. Restate the goal as concrete acceptance criteria before non-trivial edits.
2. Explore the repo with `rg` and file reads before making assumptions.
3. Prefer small, focused edits over broad refactors.
4. After edits, re-run targeted searches to catch stale call sites, duplicated
   logic, or mismatched interfaces.
5. Verify the exact scenario changed.
6. Summarize what changed, how it was verified, and any remaining gaps.

## Search

- Use `rg` for repo search; do not guess.
- Exclude generated and irrelevant directories when searching:
  `.git/`, `.idea/`, `__pycache__/`, notebook checkpoints, and similar artifacts.
- Prefer targeted searches for symbols, call sites, model names, and observable
  implementations before editing related code.

## Verification

- Use the narrowest relevant verification available for the change.
- Prefer targeted Python checks, import smoke tests, small script runs, or
  notebook-cell-level validation over broad unfocused execution.
- For Python source changes, run `ruff check` on the touched source/tests as a
  required verification step before closing the task.
- For Python source changes, run `pyright` on the touched source/tests or the
  relevant package as a required type/semantic verification step before closing
  the task.
- Validate new physics or estimator logic against exact diagonalization whenever
  the system size is small enough.
- If no automated verification exists for the changed area, say so explicitly
  and describe the manual validation you performed.
- Do not claim completion without evidence that the changed path works.
- For performance-related tasks, run a profiler as part of verification.
  Use `cProfile` plus `snakeviz` for Python-level hotspots, `scalene` for
  Python/native time and memory breakdowns, and `jax.profiler` traces for
  JAX/XLA-heavy paths.
- Do not run profilers by default for non-performance tasks.

## Reasoning Guardrails

- Match the abstraction level already established in the notebook and
  architecture document. Do not overbuild beyond the current project stage.
- Keep architecture decisions aligned with
  [architecture.md](C:\Users\balin\PycharmProjects\NQS\Balint\architecture.md)
  instead of inventing a parallel design.
- When adding new functionality, prefer extending the existing subsystem model
  instead of hard-coding one-off paths.
- Validate scientific code with the strongest available benchmark first:
  exact diagonalization, then controlled small-system numerical checks.
- If verification is weak, state the weakness clearly instead of hiding it.
