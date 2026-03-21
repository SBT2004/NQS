## T001 - Finalize `demos/exercise_1.ipynb` as a report-quality Problem 1 submission

### Goal
Make `demos/exercise_1.ipynb` fully satisfy Problem 1 of the exam brief as a polished, professor-facing report section that showcases the backend cleanly without expanding scope beyond small exact benchmarks and tiny justified additions.

### Scope
- keep the notebook centered on exact diagonalization, subsystem partitions, and exact entanglement probes
- keep the benchmark model as the 1D TFIM with open boundaries, small enough for clean exact results
- extend the current comparison from two TFIM points to three:
  - ferromagnetic `g = 0.5`
  - critical `g = 1.0`
  - paramagnetic `g = 1.5`
- preserve the current shared helpers where possible instead of introducing notebook-specific infrastructure
- keep the notebook suitable for direct inclusion in a scientific report

### Required changes
- add the ferromagnetic TFIM case at `g = 0.5` to all Exercise 1 comparisons and outputs
- replace internal-facing regime labels with report-quality labels in visible tables, figures, and prose
- expand `1/a` so it explicitly explains:
  - the spin-1/2 exact-diagonalization basis
  - the Hilbert-space size
  - how product states map to basis indices
  - what the `A|B` partition means for reshaping the statevector
- expand `1/b` so it explicitly answers:
  - what the eigenvalues of `rho_A` mean in Schmidt / MPS language
  - what entanglement scaling is expected as `|A|` changes
  - whether von Neumann entropy can be evaluated efficiently from a neural quantum state, and why not in general
- expand `1/c` so it includes:
  - the extremal purity limits
  - a formal derivation of the SWAP identity
  - a plain-language explanation of what the SWAP estimator checks
  - the `alpha > 2` replica generalization
  - why direct access to full `rho_A` or all of its eigenvalues is problematic for NQS
- sharpen `1/d` so it directly distinguishes:
  - area law
  - volume law
  - long-range entanglement
  - the different sensitivity of `S1` and `S2` to small Schmidt values
- add caption-style interpretation text after each visible table and figure
- add a short final conclusions cell that answers the exam prompts directly in polished prose
- fix broken LaTeX / control-character issues in markdown cells so all equations render correctly
- remove dev-oriented phrasing that reads like internal notes rather than report text
- keep exports limited to clean, relevant report artifacts

### Success criteria
- each subproblem `1/a` through `1/d` contains both numerical evidence and a direct prose answer
- the notebook compares ferromagnetic, critical, and paramagnetic TFIM regimes on the same footing
- all markdown renders correctly in notebook form and in exported report material
- the notebook reads as a polished report section rather than an exploratory analysis notebook
- the resulting code path remains compact, backend-driven, and clearly within exam scope

### Current gaps that keep this task open
- no ferromagnetic comparison point yet
- SWAP identity is stated but not formally derived
- markdown contains rendering/control-character issues
- several answers are still implied by outputs rather than stated explicitly in report prose
- visible labeling and output presentation are still more internal than publication-facing


## T002 - Keep Exercise 1 backend support minimal, reusable, and demonstrative

### Goal
Support the Exercise 1 notebook with only tiny, justified backend-facing improvements so the final result showcases the project architecture without drifting into refactoring for its own sake.

### Scope
- prefer reusing existing helpers in `src/nqs/observables.py` and `src/nqs/workflows/_core.py`
- allow only small helper additions if they remove obvious repetition or improve report-facing clarity
- avoid broad abstractions, notebook-specific machinery, or speculative cleanup outside Exercise 1 needs

### Required changes
- check whether any repeated Exercise 1 notebook logic should be moved into a tiny shared helper
- keep helper additions narrow and directly tied to Problem 1 deliverables
- ensure notebook code demonstrates the backend through clean use of shared APIs rather than verbose inline logic
- avoid introducing new architecture that is not needed for the final report-quality notebook

### Success criteria
- Exercise 1 notebook code stays concise and readable
- backend capabilities are visible through usage, not through excessive explanation
- any helper changes are minimal, justified, and directly useful to Exercise 1
- no broad refactor is introduced under the cover of notebook cleanup


## T003 - Clean and normalize Exercise 1 report artifacts

### Goal
Make the Exercise 1 output set consistent, report-ready, and easy to inspect.

### Scope
- focus only on `demos/report_outputs/exercise_1`
- keep artifact names, labels, and contents aligned with the final notebook narrative
- remove stale, misleading, or unrelated Exercise 1 artifacts

### Required changes
- audit the current Exercise 1 output directory for stale or unrelated files
- ensure exported filenames match the final terminology used in the notebook
- keep only report-relevant tables and figures
- ensure the final output set corresponds to the final three-regime comparison and polished report structure

### Success criteria
- the Exercise 1 output directory contains only artifacts that support the final notebook
- artifact names are consistent with the notebook’s visible terminology
- no stale training-related or mismatched files remain associated with Exercise 1
- a reviewer can inspect the output directory and immediately understand what belongs to the final report
