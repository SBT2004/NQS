## T003 - Rerun Exercise 1 demo with periodic boundary conditions

### Goal
Regenerate the Exercise 1 demo flow using periodic boundary conditions instead of the current open-boundary setup so the notebook outputs and downstream report assets reflect the PBC variant.

### Scope
- focus on `demos/exercise_1.ipynb` and any narrow workflow/config support it needs
- update only the Exercise 1 demo path from OBC to PBC
- preserve the existing notebook structure unless a small change is required to make the boundary-condition choice explicit

### Required changes
- replace the Exercise 1 demo configuration that currently uses OBC with the corresponding PBC setup
- rerun the relevant Exercise 1 notebook cells so cached outputs match the new boundary-condition choice
- ensure any saved Exercise 1 artifacts produced by the notebook are refreshed from the PBC run
- keep file/output locations consistent with the current demo and report workflow

### Success criteria
- `demos/exercise_1.ipynb` clearly runs the Exercise 1 demo with `pbc=True` rather than OBC
- the notebook executes successfully after the boundary-condition change
- refreshed Exercise 1 output artifacts exist and are consistent with the PBC configuration

### Status
Pending.


## T005 - Refactor local operators to use bit-level connected-element logic

### Goal
Refactor `operator.py` so built-in common local spin operators and terms, such as `sx` and related TFIM/J1-J2 local actions, use bit-operations and connected-element logic directly instead of dense matrix multiplication wherever that is possible, while preserving a general matrix-based path for arbitrary user-defined local matrices.

### Scope
- focus on the core operator implementation in `operator.py` and any narrow helper code needed to support fast connected-element evaluation
- optimize the built-in common spin operators/terms that map cleanly to bit flips and diagonal factors
- keep support for arbitrary user-provided local matrices as a fallback path
- do not change the public operator-builder surface unless required to route the fast path

### Required changes
- add or refactor the operator internals so common built-in local terms are evaluated via bit-level state transformations and connected matrix elements rather than constructing dense local matrices
- represent off-diagonal spin flips and diagonal contributions directly in the connected-element path for supported built-in terms
- preserve exact matrix-based handling for arbitrary user-defined local matrices, including terms that cannot be reduced to simple bit operations
- ensure the ED and operator-construction paths continue to produce the same Hamiltonians for supported models
- keep the fallback path explicit so mixed operators can combine built-in fast terms and general user-defined terms safely

### Success criteria
- built-in common spin operators/terms use the bit-level connected-element path when possible
- arbitrary user-defined local matrices still evaluate correctly through the general matrix-based fallback
- operator-based TFIM and J1-J2 workflows remain numerically unchanged for existing benchmark cases
- exact matrix construction for small-system ED remains available and consistent with the fast path

### Status
Pending.


## T004 - Add graph visualizer outputs for Exercise 1 report assets

### Goal
Create a small graph visualizer backed by an external graph library and use it in `demos/exercise_1.ipynb` to generate report-ready lattice figures for the TFIM chain and J1-J2 square lattice examples.

### Scope
- focus on `src/nqs/graph.py`, `demos/exercise_1.ipynb`, and any narrow helper module or dependency wiring needed for graph rendering
- keep the visualizer lightweight and aligned with the project's existing graph abstractions
- limit the requested report assets to the two specified figures

### Required changes
- add a small graph-visualization path that uses an external graph library for project graphs
- make the visualizer work with the structures defined in `graph.py`
- in `demos/exercise_1.ipynb`, generate one figure for a length-16 TFIM chain with blue edges
- in `demos/exercise_1.ipynb`, generate one figure for a `4x4` J1-J2 lattice with blue nearest-neighbor edges and red second-neighbor edges
- save the resulting figure images under `demos/report_outputs`

### Success criteria
- the repo contains a working graph-visualization helper that depends on an external graph library and can render the project graph objects
- `demos/exercise_1.ipynb` produces both requested figures with the specified edge-color conventions
- the generated image files are present under `demos/report_outputs` and are suitable for report inclusion

### Status
Pending.
