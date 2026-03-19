## T001 - Add an explicit sampled-SWAP execution mode for Renyi-2

### Goal
Make the Renyi-2 observable capable of demonstrating real Monte Carlo SWAP-estimator behavior even on small systems where exact-state shortcuts are available.

### Scope
- add an explicit option that forces sampled SWAP evaluation instead of exact-state fallback
- thread that option through the relevant observable helpers and callback path
- add targeted tests covering both exact and forced-sampled behavior

### Success criteria
- callers can explicitly choose sampled Renyi-2 evaluation on small systems
- the exact fallback remains available when requested
- tests verify that the sampled path is actually used when forced
- the final notebook can demonstrate noisy sampled Renyi-2 behavior instead of only exact values

### Architecture context
- observables include exact entropy tools and scalable Renyi-2 via SWAP
- the neural-quantum-state workflow should expose the scalable entanglement estimator, not hide it behind exact shortcuts

### Notes
- keep the option narrow and explicit
- do not redesign the full observables module in this task

## T002 - Add an exam-oriented architecture-comparison experiment harness

### Goal
Create a reusable experiment path for the exam’s disorder-averaged architecture comparisons instead of leaving that logic implicit in general-purpose demo helpers.

### Scope
- add a helper for repeated random initializations across RBM, FFNN, and CNN
- include subsystem-size Renyi-2 scans and parameter-count reporting for each architecture
- aggregate disorder-averaged results into notebook-ready tables or structured outputs

### Success criteria
- there is a dedicated helper for disorder-averaged architecture comparisons
- the helper reports both entanglement results and parameter-count metadata
- targeted tests or checks cover the core aggregation path
- the codebase moves closer to the exam workflow instead of only providing generic demos

### Architecture context
- scientific deliverables include architecture comparisons and entropy-versus-subsystem-size studies
- notebook-facing workflows should stay close to the intended analysis surface

### Notes
- keep the first version focused on untrained/random-initialization studies
- avoid mixing in large plotting code here

## T003 - Add an exam-oriented Hamiltonian/system-size sweep harness

### Goal
Support the exam’s critical-versus-away-from-critical and larger-system comparison studies through a dedicated experiment workflow.

### Scope
- add a helper for sweeping Hamiltonian parameters and system sizes for TFIM and J1-J2
- support tracking energy and Renyi-2 during training for the selected runs
- return structured outputs suitable for critical-vs-noncritical comparisons

### Success criteria
- there is a dedicated helper for Hamiltonian-parameter and system-size sweeps
- the helper can record Renyi-2 during training for the selected runs
- targeted checks cover at least one small controlled sweep path
- the repository is closer to the final exam workflow for Problem 3-style comparisons

### Architecture context
- validation and scientific deliverables include multiple models, multiple system sizes, and training-time entropy behavior
- callbacks and observables should support training-loop analysis without ad hoc notebook logic

### Notes
- keep the first version focused on one reviewable sweep path
- defer heavy plotting and long-run notebook polish to the final notebook task

## T004 - Add a GHZ-state training bonus workflow

### Goal
Cover the exam’s GHZ-style bonus path with a dedicated, notebook-ready training workflow instead of leaving it as an implied extension.

### Scope
- add a small GHZ target-state preparation or training workflow under the existing project abstractions. The GHZ state is just the special case of the TFIM with h=0
- include the narrow observable/evaluation path needed to judge success
- keep the implementation limited to one bonus workflow and its direct checks

### Success criteria
- there is a project-owned GHZ-focused training or evaluation path
- the path has at least one targeted test or verification check
- the workflow is accessible to the final notebook without bespoke ad hoc code
- the repository moves closer to the complete exam coverage described by the project goal

### Architecture context
- future/bonus scientific workflows should still sit on top of the shared model, sampler, variational-state, and observables abstractions

### Notes
- keep this intentionally narrow
- do not broaden it into a generic target-state training framework unless required by the implementation

## T005 - Build one integrated final-exam notebook and report-output layer

### Goal
Provide a single exam-oriented notebook that ties together benchmarks, observables, training studies, and figure/report outputs in the form needed to solve the exercises.

### Scope
- add one integrated notebook under `demos/` that covers the required exam workflow
- include sectioned outputs for ED benchmarks, entropy scans, architecture comparisons, training-time Renyi-2, and larger-system studies
- add the minimal helper layer needed for consistent figure generation and report-ready outputs

### Success criteria
- the repository contains one integrated exam-oriented notebook rather than only separate showcase/benchmark notebooks
- the notebook exercises the implemented helper paths instead of duplicating core logic inline
- figures/tables required by the exam workflow are generated through a reusable helper path where needed
- the result is structurally aligned with `architecture.md` and materially closer to a final submission workflow

### Architecture context
- notebook-derived target usage
- scientific deliverables include benchmarks, entropy scans, architecture comparisons, and training-time observables

### Notes
- keep notebook code orchestration-focused; shared logic should live in modules/helpers
- this task should build on the earlier observable and experiment-harness tasks rather than replace them
