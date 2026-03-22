## T001 - Replace dense ED with a sparse ground-state solver

### Goal
Replace the current dense exact-diagonalization path with a sparse, ground-state-focused implementation that never builds a dense Hamiltonian in the main ED flow.

### Scope
- focus on `src/nqs/exact_diag.py` and the operator-side support it needs
- optimize for ground-state energy and ground-state vector only
- use sparse matrices and iterative Hermitian eigensolvers
- do not introduce symmetry sectors, momentum blocks, or any other symmetry reduction
- keep any dense Hamiltonian utilities separate from the production ED path

### Required changes
- add a project-owned sparse Hamiltonian construction path for `nqs.operator.Operator`
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

