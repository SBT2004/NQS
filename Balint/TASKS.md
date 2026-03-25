## T005 - Strengthen Exercise 2 report support material

### Goal
Close the remaining evidence gaps for the Exercise 2 final report so the qualitative conclusions about sampling, SWAP estimation, initialization, and architecture dependence are supported to a higher standard, and so the current FFNN/CNN results are either validated or explained.

### Scope
- focus on the Exercise 2 retained report material and any future reruns or export updates needed to support the written claims
- cover estimator validation, sampler behavior, initialization sensitivity, and architecture comparison
- do not mix this report follow-up with unrelated optimization or refactor tasks

### Checklist
- add a clearer subsystem-size versus noise study for the SWAP estimator so the discussion of estimator degradation is supported by explicit retained or regenerated evidence
- assess whether the current small-system SWAP validation is sufficient for the final report or whether a stronger benchmark should be added
- add a sampler-mixing or autocorrelation analysis for the single-spin local update, especially near the critical point, so the discussion of critical slowdown is backed by measured data rather than only by physical expectation
- separate estimator failure from genuine entanglement differences more explicitly in the initialization study, especially for unrestricted random complex phases
- investigate whether the very small FFNN/CNN entropies reflect genuine model behavior, estimator failure, or an implementation bug before using the architecture comparison as evidence in the final report

### Status
Partial.

### Completion Notes
- The retained SWAP-validation rerun now exists with subsystem-by-subsystem exact-versus-sampled comparisons and 95% confidence intervals in `demos/report_outputs/exercise_2/exercise_2_swap_validation.csv`, so the subsystem-size noise discussion is better supported.
- The task is still not complete because the retained support material does not export the sampler-mixing or autocorrelation evidence promised in the report path, the initialization follow-up still does not add an exact small-system benchmark that cleanly separates estimator failure from genuine entanglement changes for unrestricted complex phases, and the architecture comparison rerun still lacks the retained FFNN/CNN sanity diagnostics needed to decide whether the near-zero entropies are physical, noisy, or buggy.


## T007 - Rerun and debug the Exercise 2 architecture comparison

### Goal
Repeat the retained architecture comparison under a cleaner and lower-noise setup, and determine whether the current FFNN/CNN results are physically plausible or indicate an implementation or measurement problem.

### Scope
- focus on the Exercise 2 architecture-comparison workflow and the model-specific paths for RBM, FFNN, and CNN
- include both reruns and debugging checks needed to trust the comparison
- do not treat the current architecture ordering as final until these checks are complete

### Checklist
- rerun the architecture comparison with a larger sampling budget, more independent seeds, and explicit uncertainty estimates
- match parameter counts or model capacity more carefully across RBM, FFNN, and CNN so the comparison is not dominated by an avoidable capacity mismatch
- verify that FFNN and CNN log-amplitude outputs vary nontrivially over sampled configurations and are not collapsing to a nearly constant state
- inspect the sampled SWAP-estimator inputs for FFNN and CNN to check whether the near-zero entropies come from model behavior, poor mixing, or numerical instability
- compare acceptance, autocorrelation, and effective sample quality across the three model families under the rerun setup
- regenerate the architecture summary CSV and figure only after the above checks pass, and update the report wording accordingly

### Status
Pending.
