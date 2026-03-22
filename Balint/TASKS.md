## T002 - Add verification and microbenchmarks for sparse ED performance and correctness

### Goal
Add focused verification and benchmark coverage that proves the sparse ED refactor is correct and removes the dominant dense-memory bottleneck.

### Scope
- focus on tests and microbenchmarks for ED, sparse operator assembly, and exact expectation evaluation
- validate against small-system dense/reference results where comparison is feasible
- avoid broad benchmarking outside the ED-related surfaces changed here

### Required changes
- add correctness tests comparing sparse ED ground energies against current dense or NetKet-backed references on small systems
- add tests that compare sparse exact expectations against the previous dense expectation results on small systems
- add tests that confirm exact observable outputs derived from the sparse ED ground-state vector remain correct
- extend the core microbenchmarks to include sparse matrix assembly and sparse ground-state solve timings
- include at least one benchmark or validation case that demonstrates the main ED path no longer performs dense Hamiltonian allocation
- benchmark the open-chain TFIM Exercise 1 ED path incrementally at chain lengths `6, 8, 10, 12, 14, 16`, changing the code, running, increasing the size, and running again rather than implementing a full notebook sweep upfront
- record per-size runtime and whether each run completes without memory or solver failure
- use the 16-spin case as the practical optimization gate for the intended machine

### Success criteria
- sparse ED correctness is covered by automated tests
- sparse exact expectation correctness is covered by automated tests
- benchmark coverage distinguishes sparse assembly cost from sparse solve cost
- verification provides evidence that the dense-memory bottleneck has been removed from the main ED path
- the incremental Exercise 1 ED benchmark reaches the 16-spin open-chain TFIM case without memory or solver issues
- the 16-spin Exercise 1 ED case completes in under one minute; if it does, the optimization is considered successful

### Status
Partial.

Verification notes:
- Automated correctness coverage now exists for sparse ED and sparse exact expectations: `tests/test_operator.py`, `tests/test_vmc.py`, and `tests/test_notebook_workflows.py` compare sparse results against dense small-system references and assert the main exact-observables path does not densify the Hamiltonian.
- `tests/test_core_microbenchmarks.py` now distinguishes `exact_diag.sparse_operator_matrix` assembly from `exact_diag.solve_sparse_ground_state` solve cost, and `src/nqs/workflows/_core.py` exposes `run_incremental_exercise_1_ed_benchmark(...)` with per-size assembly, solve, runtime, completion, and failure fields.
- The required incremental Exercise 1 gate is still only partially covered in repo state: automated checks call the helper only for `[4, 6]`, and there is no saved `6, 8, 10, 12, 14, 16` runtime/completion artifact from that incremental benchmark path.
- `demos/exercise_1.ipynb` and `demos/report_outputs/exercise_1/exercise_1_exact_summary.csv` do show open-chain `L=16` exact-observable output, but they do not record the required per-size assembly/solve/runtime measurements or solver-failure status.
- There is still no repo evidence that the 16-spin Exercise 1 ED case completes without memory or solver issues in under one minute on the intended machine.
## T003 - JIT the local Metropolis sampler loop for larger TFIM VMC runs

### Goal
Remove Python-step overhead from the local Metropolis sampler so warmed `5x5` TFIM VMC runs spend materially less time in per-step control flow and repeated dispatch.

### Scope
- focus on `src/nqs/sampler.py` and the narrow workflow/tests it needs
- keep the public sampler configuration and notebook-facing API stable
- preserve the current local single-spin-flip proposal semantics
- do not broaden scope into new sampler families such as exchange or cluster updates

### Required changes
- replace the Python loops in sampler draw/thermalization paths with JAX-friendly control flow such as `lax.scan`
- keep chain persistence behavior compatible with the current `sample(...)` and `independent_sample(...)` contract
- preserve batched chain updates and proposal acceptance semantics
- ensure the refactor works for the current RBM, FFNN, and CNN interfaces without sampler-side model special cases

### Success criteria
- warmed sampling no longer spends the dominant control-flow cost in Python loops for the current local Metropolis path
- `sample(...)` and `independent_sample(...)` remain numerically compatible with the current sampler semantics
- the notebook-facing VMC workflow continues to run unchanged on the refactored sampler

### Status
Not complete.

Verification notes:
- `src/nqs/sampler.py` still performs thermalization and sample collection in explicit Python `for` loops inside `_draw_samples(...)`.
- `_metropolis_step(...)` is still dispatched once per thermalization/collection step from Python rather than being fused under `jax.lax.scan` or similar JAX control flow.
- Current tests exercise correctness and notebook compatibility, but the repo does not yet contain a measured `5x5` warmed benchmark artifact showing reduced sampler control-flow cost.


## T004 - Vectorize local-energy assembly for sampled VMC expectations

### Goal
Reduce the Python and host-side overhead in sampled VMC expectation evaluation by replacing per-sample local-energy assembly loops with batched project-owned logic.

### Scope
- focus on `src/nqs/expectation.py` and any narrow operator-side support it needs
- keep the existing VMC gradient estimator contract intact
- optimize only the sampled local-energy path, not the exact-state branch targeted by earlier sparse-ED tasks

### Required changes
- remove or substantially reduce the Python loop over samples in `_local_energies(...)`
- batch connected-state evaluation and model log-amplitude queries where feasible
- preserve compatibility with the current operator connected-elements contract
- keep the resulting energy and gradient estimates numerically consistent with the current sampled implementation within Monte Carlo tolerance

### Success criteria
- warmed profiler runs show materially less time in sampled local-energy assembly on the `5x5` TFIM benchmark path
- sampled VMC energies and gradients remain consistent with the current implementation on controlled small-system checks
- the public variational-state and driver APIs remain unchanged

### Status
Not complete.

Verification notes:
- `src/nqs/expectation.py` `_local_energies(...)` still converts sampled states to NumPy, iterates over samples in Python, and iterates over connected states per sample before re-entering `model.log_psi(...)`.
- `tests/test_vmc.py` covers sampled local-energy correctness, but the repo does not yet contain a measured `5x5` benchmark artifact showing materially lower warmed local-energy assembly cost.


## T005 - Eliminate repeated `log_psi` evaluation across VMC hot paths

### Goal
Lower the dominant model-evaluation cost in larger-system VMC by reusing or fusing repeated `log_psi` work across sampling, gradient, and benchmark-diagnostic paths.

### Scope
- focus on `src/nqs/models.py`, `src/nqs/expectation.py`, `src/nqs/sampler.py`, and narrowly related workflow code
- keep the shared model interface stable for RBM, FFNN, and CNN
- avoid introducing architecture-specific caches in shared infrastructure unless the reuse is interface-driven

### Required changes
- identify repeated `log_psi` evaluations within the current VMC step and benchmark callbacks
- reuse already-available model evaluations where the same states are queried multiple times
- avoid recomputing identical batches across sampler acceptance, local-energy evaluation, and benchmark diagnostics when a safe shared value is available
- keep parameter/state invalidation rules explicit so reused values cannot silently become stale

### Success criteria
- warmed profiler runs show a measurable reduction in cumulative time under model `log_psi/apply` during the `5x5` TFIM benchmark path
- reuse does not change the notebook-facing VMC workflow or require ansatz-specific special cases
- any new cache or fused path remains correct under parameter updates and sampler-state updates

### Status
Not complete.

Verification notes:
- `src/nqs/sampler.py`, `src/nqs/expectation.py`, and `src/nqs/models.py` still call `model.log_psi(...)` independently across sampler acceptance, sampled local-energy evaluation, and gradient code paths.
- There is no shared cache or fused evaluation path with explicit invalidation tied to parameter or chain-state changes.
- The repo does not yet contain a measured `5x5` before/after artifact showing reduced cumulative `log_psi/apply` cost.


## T006 - Split training-time and report-time cost metrics for non-ED benchmarks

### Goal
Make the larger-system non-ED benchmark outputs distinguish optimizer cost from diagnostic/report generation cost so architecture tradeoffs remain interpretable on `5x5` TFIM runs.

### Scope
- focus on `src/nqs/workflows/_core.py`, `demos/exercise_2.ipynb`, and narrow tests
- keep the current non-ED benchmark workflow notebook-facing
- do not broaden scope into new benchmarking frameworks outside the existing workflow helper surface

### Required changes
- define explicit benchmark metrics for at least training runtime and full report/runtime cost
- keep callback, post-processing, and optional entropy-scan costs visible rather than folding them into one ambiguous column
- update the Exercise 2 notebook narrative and plots to use the clearer runtime split
- ensure the benchmark remains sampled/non-ED in semantics for all reported rows

### Success criteria
- the Exercise 2 non-ED benchmark clearly distinguishes optimizer cost from diagnostic/reporting cost
- notebook plots and tables no longer imply that a partial timing column represents the full benchmark cost
- the resulting runtime comparison remains aligned with the actual reported outputs for each benchmark row

### Status
Not complete.

Verification notes:
- `src/nqs/workflows/_core.py` now reports `runtime_seconds`, `callback_runtime_seconds`, and `total_runtime_seconds` from `run_non_ed_vmc_benchmark(...)`.
- `demos/exercise_2.ipynb` still selects and plots `runtime_seconds` as the benchmark cost column, so the notebook-facing analysis does not yet clearly separate optimizer time from report/diagnostic time.
- `demos/report_outputs/exercise_2/` does not contain a saved large-benchmark summary/report artifact that exposes the split runtime columns.


## T007 - Add a dedicated `5x5` TFIM VMC performance regression benchmark

### Goal
Turn the current profiler-guided `5x5` TFIM VMC investigation into a repeatable benchmark/regression surface so future sampler and expectation changes can be evaluated against a concrete larger-system target.

### Scope
- focus on tests, microbenchmarks, and lightweight profiling hooks for the `5x5` TFIM VMC path
- keep the benchmark narrow and project-owned
- avoid broad notebook execution sweeps as the primary regression signal

### Required changes
- add a dedicated benchmark or profiling harness for the warmed `5x5` TFIM VMC path
- record at minimum model-evaluation cost, sampler cost, local-energy/gradient cost, and end-to-end benchmark timing
- separate cold-start compile/tracing effects from warmed execution measurements
- define a practical regression gate or reporting format that can be rerun after performance-oriented refactors

### Success criteria
- the repo contains a repeatable `5x5` TFIM VMC performance benchmark rather than only one-off profiler notes
- the benchmark distinguishes cold-start and warmed behavior
- future optimization work on sampler, expectation, or model evaluation can be judged against a stable measured target

### Status
Not complete.

Verification notes:
- There is no dedicated project-owned `5x5` TFIM performance harness in `tests/` or `src/nqs/workflows/`; the available non-ED benchmark helper is a general training/report helper rather than a narrow regression benchmark.
- The current repo does not record per-component warmed cost for model evaluation, sampler, and local-energy/gradient work on the `5x5` path.
- There is no cold-start versus warmed execution separation or practical rerunnable regression gate for the `5x5` TFIM VMC target.
