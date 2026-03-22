## T001 - Replace dense ED with a sparse ground-state solver

### Goal
Replace the current dense exact-diagonalization path with a sparse, ground-state-focused implementation that never builds a dense Hamiltonian in the main ED flow.

### Scope
- focus on `src/nqs/exact_diag.py` and the operator-side support it needs
- optimize for ground-state energy and ground-state vector only
- use sparse matrices and iterative Hermitian eigensolvers (Lanczos)
- do not introduce symmetry sectors, momentum blocks, or any other symmetry reduction
- keep any dense Hamiltonian utilities separate from the production ED path

### Required changes
- add a project-owned sparse Hamiltonian construction path for `nqs.operator.Operator`
- The construction should go as follows: the operator module outputs all nonzero matrix elements for going from one sigma to another, that is one row of the matrix
- assemble the Hamiltonian as a SciPy sparse Hermitian matrix, preferably CSR after COO-style accumulation
- implement the main ED solver with a sparse Hermitian eigensolver such as `scipy.sparse.linalg.eigsh`
- return only the lowest eigenpair on the main ED path
- remove dense-matrix construction from the default ED implementation
- keep any dense matrix helper explicitly demo/debug-only rather than part of the main solver path
- ensure the returned ground-state vector remains suitable for downstream exact observables such as reduced density matrices and von Neumann entropy

### Success criteria
- the main ED path no longer allocates a dense Hamiltonian
- exact ground-state energy and ground-state vector remain available through the project-owned ED API
- small-system ground-state energies match the current dense or reference results within numerical tolerance
- the ED path scales past the current dense-memory failure mode without relying on symmetries

### Status
Not complete.

Verification notes:
- The current repo state still uses dense ED in `src/nqs/exact_diag.py`: `operator_matrix(...)` allocates a full dense Hamiltonian and `exact_ground_state(...)` calls `np.linalg.eigh(...)`.
- No project-owned sparse Hamiltonian construction path or `eigsh`-based ground-state solver is present yet.


## T002 - Decouple exact expectations from dense Hamiltonian materialization

### Goal
Stop using dense Hamiltonian materialization for exact small-system expectation values so the exact expectation backend no longer shares the same memory bottleneck as dense ED.

### Scope
- focus on `src/nqs/expectation.py`
- keep the current exact statevector logic where it is still appropriate
- replace only the dense operator application path
- do not change the sampled expectation path

### Required changes
- remove the dependency on dense `operator_matrix(...)` from the exact expectation branch
- compute exact energies via sparse operator application to a dense statevector
- preserve the current exact/sampled backend selection behavior
- keep the exact expectation result numerically consistent with the previous implementation on small systems

### Success criteria
- exact expectation evaluation no longer builds a dense Hamiltonian
- exact energy values agree with the previous small-system implementation within tolerance
- the exact expectation backend remains compatible with the current variational-state and driver code

### Status
Not complete.

Verification notes:
- The exact expectation branch in `src/nqs/expectation.py` still calls `operator_matrix(...)` and multiplies by a dense Hamiltonian inside `_exact_expectation_mean(...)`.
- The sampled expectation path is separate, but the exact path has not been migrated to sparse operator application.


## T003 - Split production ED APIs from dense demo/debug helpers

### Goal
Make the exact-diagonalization API sparse-first and ground-state-focused, while isolating any dense matrix helpers as explicit demo/debug utilities rather than part of the production runtime path.

### Scope
- focus on the public-facing ED helpers and their immediate callers
- preserve support for dense-statevector-based exact observables
- do not keep full-spectrum dense return data on the main ED path
- limit dense helpers to explicit demo/debug use

### Required changes
- redefine the main ED result contract around `ground_energy` and `ground_state`
- remove assumptions that the main ED API returns a dense Hamiltonian or full eigenvalue list
- update project-owned callers to consume the sparse-first ground-state result shape
- keep any dense helper naming and placement explicit enough that it is not confused with the production ED path

### Success criteria
- the main ED API no longer implies dense Hamiltonian ownership
- downstream workflows continue to work with the sparse-first result shape
- dense helpers, if retained, are clearly separated from the production ED path

### Status
Not complete.

Verification notes:
- The production ED API is still dense-oriented: `ExactDiagResult` includes `matrix` and `eigenvalues`, and `exact_ground_state(...)` returns both.
- Dense helpers have not been separated from the main ED runtime path yet.


## T004 - Preserve exact-observable workflows on top of sparse ED ground states

### Goal
Keep exact report and analysis workflows working after the sparse ED refactor by ensuring they consume the sparse-solver ground-state vector rather than any dense Hamiltonian artifacts.

### Scope
- focus on exact-observable and workflow code that depends on `exact_ground_state(...)`
- preserve current reduced-density-matrix, entropy, and entanglement-spectrum behavior
- do not expand scope into new observables or new physics functionality

### Required changes
- update exact workflow helpers to rely only on `ground_energy` and `ground_state`
- preserve reduced density matrix, von Neumann entropy, Renyi-from-statevector, and entanglement spectrum calculations
- confirm no workflow still depends on dense ED outputs such as full matrices or full spectra
- keep notebook-facing exact benchmark helpers report-friendly after the refactor

### Success criteria
- exact-observable workflows continue to produce the same report-facing results on small systems
- downstream exact entropy calculations still operate from the returned dense ground-state vector
- no production exact workflow path depends on dense Hamiltonian materialization

### Status
Not complete.

Verification notes:
- The exact-observable workflows still rely on the current dense ED result shape because the sparse ground-state API refactor has not happened yet.
- Reduced-density-matrix and entropy code still works on the returned dense statevector, but no workflow migration to a sparse-first ED contract has been implemented.


## T005 - Add verification and microbenchmarks for sparse ED performance and correctness

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
- benchmark the open-chain TFIM Exercise 1 ED path incrementally at chain lengths `6, 8, 10, 12, 14, 16, 18, 20`, changing the code, running, increasing the size, and running again rather than implementing a full notebook sweep upfront
- record per-size runtime and whether each run completes without memory or solver failure
- use the 20-spin case as the practical optimization gate for the intended machine

### Success criteria
- sparse ED correctness is covered by automated tests
- sparse exact expectation correctness is covered by automated tests
- benchmark coverage distinguishes sparse assembly cost from sparse solve cost
- verification provides evidence that the dense-memory bottleneck has been removed from the main ED path
- the incremental Exercise 1 ED benchmark reaches the 20-spin open-chain TFIM case without memory or solver issues
- the 20-spin Exercise 1 ED case completes in under one minute; if it does, the optimization is considered successful

### Status
Not complete.

Verification notes:
- Existing tests and microbenchmarks still target the dense ED path; `tests/test_core_microbenchmarks.py` benchmarks `exact_diag.operator_matrix` rather than sparse assembly or sparse ground-state solves.
- The incremental `6, 8, 10, 12, 14, 16, 18, 20` Exercise 1 benchmark gate has not been implemented or verified, and there is no evidence yet that the 20-spin case completes under one minute without issues.
## T006 - JIT the local Metropolis sampler loop for larger TFIM VMC runs

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
- Recent profiler evidence on the `5x5` TFIM non-ED benchmark still shows `src/nqs/sampler.py` `_draw_samples` and `_metropolis_step` among the dominant warmed-path costs.
- The current sampler implementation still performs thermalization and collection in explicit Python loops.


## T007 - Vectorize local-energy assembly for sampled VMC expectations

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
- Recent profiler evidence on the `5x5` TFIM non-ED benchmark still shows `src/nqs/expectation.py` `expect_and_grad(...)` and `_local_energies(...)` as major warmed-path costs.
- The current local-energy path still loops over samples and connected states in Python while repeatedly re-entering model evaluation.


## T008 - Eliminate repeated `log_psi` evaluation across VMC hot paths

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
- Recent warmed `5x5` TFIM profiles still show `src/nqs/models.py` `log_psi(...)` and `apply(...)` as the dominant cumulative cost.
- The current implementation still evaluates model log-amplitudes repeatedly across sampler, gradient, and diagnostic code paths.


## T009 - Split training-time and report-time cost metrics for non-ED benchmarks

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
- The current non-ED benchmark helper now reports `runtime_seconds`, `callback_runtime_seconds`, and `total_runtime_seconds`, but the notebook/report surface has not yet been reworked into a clearly separated training-vs-report benchmark analysis.
- The current per-step history still mixes pre-update energy with callback observables, which is acceptable for the present task but leaves the benchmark timing/reporting surface worth tightening further.


## T010 - Add a dedicated `5x5` TFIM VMC performance regression benchmark

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
- The previous report identified `5x5` TFIM as the practical optimization target, but there is not yet a dedicated repeatable benchmark artifact or regression gate for that path.
- Existing workflow tests validate behavior, not sustained larger-system performance characteristics.
