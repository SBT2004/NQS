# Architecture: JAX NetKet-Replacement For Neural Quantum States

## Project Goal

This project aims to reproduce the high-level workflow used in
[Netket.ipynb](C:\Users\balin\PycharmProjects\NQS\Balint\demos\Netket.ipynb)
without depending on NetKet at runtime. NetKet is the reference for API shape
and user experience, but the implementation should be project-owned and built
with JAX.

The scientific target is the exam brief in
[exam.pdf](C:\Users\balin\Documents\MSc\1 sem\CompPhys\NQS_Project\exam.pdf),
specialized to:

- 2D transverse-field Ising model (TFIM) on square lattices with periodic
  boundary conditions
- 2D J1-J2 model on square lattices with periodic boundary conditions
- RBM, FFNN, and CNN neural quantum state architectures

The intended user flow mirrors the notebook:

`graph -> hilbert -> operator/hamiltonian -> model -> sampler -> variational state -> optimizer/driver -> observables`

## Scope

### In Scope

- A NetKet-like high-level Python API implemented with JAX-compatible project
  code
- Lattice and Hilbert-space abstractions for spin-1/2 systems
- Hamiltonian builders for 2D TFIM and 2D J1-J2
- Variational wavefunction models: RBM, FFNN, CNN
- Monte Carlo sampling with at least a local Metropolis sampler
- Variational Monte Carlo training
- Exact diagonalization for small-system benchmarks
- Entanglement observables needed by the exam:
  - exact von Neumann entropy for small systems
  - Renyi-2 entropy from a SWAP estimator for neural quantum states
- Plot-ready outputs for energy, entropy, and subsystem-size scans

### Out Of Scope For The First Implementation

- Full NetKet feature parity
- Distributed execution, MPI support, or multi-host sampling
- General symmetry-aware neural architectures
- Automated convergence heuristics based on autocorrelation analysis
- Large-scale production benchmarking

## Design Principles

- Match the notebook's abstraction level, not NetKet's internal complexity.
- Keep the public API small and explicit.
- Separate physics definitions from sampling and training logic.
- Make models swappable: the sampler and training loop must not depend on a
  specific ansatz.
- Use exact diagonalization as the correctness anchor for small lattices.
- Prefer batched JAX functions over per-sample Python loops in model and
  estimator code.

## Reference API Surface

The project should borrow the conceptual surface of the following NetKet APIs:

- [Hypercube](https://netket.readthedocs.io/en/latest/api/_generated/graph/netket.graph.Hypercube.html)
- [Spin](https://netket.readthedocs.io/en/latest/api/_generated/hilbert/netket.hilbert.Spin.html)
- [RBM](https://netket.readthedocs.io/en/v3.15.1/api/_generated/models/netket.models.RBM.html)
- [Sampler docs](https://netket.readthedocs.io/en/latest/docs/sampler.html)
- [MCState](https://netket.readthedocs.io/en/latest/api/_generated/vqs/netket.vqs.MCState.html)
- [VMC](https://netket.readthedocs.io/en/latest/api/_generated/driver/netket.driver.VMC.html)
- [IsingJax](https://netket.readthedocs.io/en/latest/api/_generated/operator/netket.operator.IsingJax.html)
- [LocalOperator](https://netket.readthedocs.io/en/v3.6/api/_generated/operator/netket.operator.LocalOperator.html)

This is a reference list for interface design, not a mandate to copy internal
implementation details.

## Notebook-Derived Target Usage

The current notebook expresses the desired user experience:

```python
import nqs

L = 4
J = 1.0
h = 1.0
n_iter = 300

g = nqs.graph.SquareLattice(length=L, n_dim=2, pbc=True)
hi = nqs.hilbert.SpinHalf(N=g.n_nodes)

H = nqs.operator.tfim(hi=hi, graph=g, J=J, h=h)

model = nqs.models.RBM(alpha=2, param_dtype=float)
sampler = nqs.sampler.MetropolisLocal(hi, n_chains=256)

vstate = nqs.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=512,
    n_discard_per_chain=100,
)

opt = nqs.optimizer.Adam(learning_rate=1e-2)
driver = nqs.driver.VMC(H, opt, variational_state=vstate)
driver.run(
    n_iter=n_iter,
    callback=nqs.observables.entropy_callback(subsystem="half"),
    step_size=1,
)
```

This example defines the intended API shape. Exact class and module names may
change slightly during implementation, but the user-facing style should remain
at this level.

## Core Subsystems

### `graph`

Responsibility:

- Represent the square-lattice geometry and site ordering used everywhere else.

Required capabilities:

- Construct 2D square lattices with periodic boundary conditions
- Provide `n_nodes`
- Provide nearest-neighbor edge lists
- Provide next-nearest-neighbor edge lists for J1-J2
- Keep a deterministic site ordering for basis encoding, Monte Carlo updates,
  and subsystem partitions
- Optionally expose neighbor lookup helpers for diagnostics or visualization

Notes:

- The notebook already assumes the graph defines the ordering and adjacency.
- The J1-J2 implementation should distinguish J1 bonds from diagonal J2 bonds.

### `hilbert`

Responsibility:

- Represent the spin-1/2 Hilbert space independently from the Hamiltonian or
  model.

Required capabilities:

- Batch representation of spin configurations
- Encoding and decoding between spin arrays and basis indices
- Enumeration of all basis states for exact diagonalization only
- A clear convention for spin values, for example `{-1, +1}` or
  `{-1/2, +1/2}`, used consistently across the project

Notes:

- The notebook currently reconstructs statevectors and reduced density matrices
  from enumerated states. That should remain possible for small-system ED and
  validation.

### `operator`

Responsibility:

- Define linear maps on the Hilbert space as sums of local terms.

Required capabilities:

- A generic operator abstraction built from local matrices acting on ordered
  site tuples
- A `LocalTerm`-like representation with `(sites, matrix, coefficient)`
- A core primitive that returns connected matrix elements
  `H_{sigma', sigma}` for a given basis state `sigma`
- Support for exact matrix construction on small systems for ED benchmarks
- Builder utilities that later translate graphs into TFIM and J1-J2 operator
  terms

Model-specific expectations:

- TFIM and J1-J2 belong in builder utilities layered on top of the core
  operator abstraction.

Notes:

- The operator core depends on `hilbert`, not on `graph`.
- Graphs are used one layer up to generate model-specific collections of local
  terms.
- The long-term shared primitive for both ED and NQS is the set of connected
  matrix elements `H_{sigma', sigma}`, not expectation values directly.

### `models`

Responsibility:

- Represent variational wavefunctions.

Required shared interface:

- A JAX-compatible batched forward method from configurations to log-amplitudes
  or amplitudes
- Parameter initialization from a PRNG key
- A model-independent way for `vqs` and the sampler to evaluate `logpsi(sigma)`

Required architectures:

- `RBM`: first reference ansatz and the first model to support end-to-end
  training
- `FFNN`: generic dense baseline
- `CNN`: locality-aware architecture aligned with 2D lattice structure

Notes:

- The notebook currently starts from `RBM(alpha=2)`. RBM should therefore be
  the first complete implementation.
- Complex-valued parameters are not a first requirement, but the interface
  should not make them impossible later.
- In the current VMC phase, model architectures are project-owned JAX models,
  even when temporary NetKet backends are used elsewhere in the stack.

### `sampler`

Responsibility:

- Generate samples from the variational state using ratios of probabilities or
  amplitudes.

Required capabilities:

- `MetropolisLocal` as the first sampler
- Model-agnostic operation: only the state evaluation interface should matter
- Configurable number of chains, thermalization steps, and retained samples
- JAX-friendly batching where practical

Notes:

- Sampling should remain decoupled from the model architecture so the same
  sampler works for RBM, FFNN, and CNN.

### `vqs`

Responsibility:

- Hold the model, parameters, sampler, and sampling configuration in one
  variational state object.

Required capabilities:

- Own model parameters and sampler state
- Expose batched log-amplitude evaluation
- Expose sample generation
- Expose expectation-value helpers
- Store run-level settings such as `n_samples` and `n_discard_per_chain`

Notes:

- This should be the main integration point between models, samplers, and
  observables.
- In the current implementation phase, `vqs` is still project-owned even if it
  delegates temporary sampling and energy-evaluation work to NetKet adapters.

### `driver` / `training`

Responsibility:

- Execute optimization loops for VMC.

Required capabilities:

- VMC driver that updates model parameters by minimizing energy
- Optimizer hookup, at minimum Adam
- Logging of training observables
- Callback support so entropy can be tracked during training
- A placeholder extension point for stochastic reconfiguration

Notes:

- The notebook already uses a callback to compute entropy during optimization.
- The training loop should support the same pattern without hard-coding entropy
  into the driver.
- In the current implementation phase, gradients and parameter updates are
  computed in project code with JAX autodiff and a project optimizer wrapper.

### `observables`

Responsibility:

- Provide energy and entanglement diagnostics for both ED and NQS workflows.

Required capabilities:

- Energy estimators
- Magnetization or simple correlation observables as needed for diagnostics
- Exact reduced density matrix construction for small systems
- Exact von Neumann entropy from the reduced density matrix
- Renyi-2 entropy from a SWAP estimator for sampled neural quantum states
- Training-time callback utilities for logging entropy

Important constraint:

- Von Neumann entropy is an ED or full-state small-system tool. It should not
  be treated as the default scalable NQS observable.
- Renyi-2 via SWAP is the main scalable entanglement probe for the neural
  quantum state part of the project.

### `ed`

Responsibility:

- Provide exact diagonalization for small systems as a benchmark subsystem.

Required capabilities:

- Build exact Hamiltonian matrices for the supported models
- Compute exact ground states and reference energies
- Support subsystem partitions for entropy evaluation
- Benchmark VMC energy and entropy trends on small lattices

Notes:

- ED is required, not optional. It is the correctness anchor for the project
  and directly supports the exam deliverables.

## Scientific Deliverables

The codebase should support the following exam-relevant outcomes:

- Exact ground-state benchmarks on small systems
- Comparison of VMC energy against ED where ED is feasible
- Renyi-2 entropy versus subsystem size
- Entropy behavior during training
- Comparison of architectures: RBM, FFNN, CNN
- Comparison of models: 2D TFIM and 2D J1-J2
- Report-ready plots and logged observables

## Minimum Implementation Order

1. Graph and Hilbert-space core for spin-1/2 square lattices
2. TFIM operator and exact diagonalization benchmark path
3. RBM model, local Metropolis sampler, and basic VMC training
4. Renyi-2 SWAP estimator and entropy logging callback
5. FFNN and CNN support under the same variational-state interface
6. J1-J2 Hamiltonian support
7. Larger-system experiments and architecture comparisons

This order keeps the first milestone small but scientifically testable.

## Validation Requirements

The first complete version should be able to answer all of the following:

- Does 2D TFIM VMC recover reasonable small-system energies compared with ED?
- Does the same user-facing workflow work when swapping RBM for FFNN or CNN?
- Does the Renyi-2 estimator behave sensibly as subsystem size increases?
- Can entropy be tracked during training without breaking the VMC loop?
- Does the code support both TFIM and J1-J2 without rewriting the training
  stack?

## Expected Contributor Behavior

When extending the codebase, prefer the following:

- Keep user-facing code notebook-friendly and close to the NetKet-inspired
  abstraction level
- Add new physics models as operator builders, not by special-casing the driver
- Add new ansatz classes behind the shared model interface
- Add new observables through estimator utilities and callbacks
- Validate new functionality against ED whenever system size allows

## Future Extensions

These are good follow-up targets, but should not block the initial
implementation:

- Stochastic reconfiguration
- Exchange or cluster samplers
- Complex-valued wavefunctions
- Additional entanglement probes for small systems
- Symmetry-aware architectures
- Better convergence diagnostics and adaptive stopping criteria

## Summary

This repository should become a JAX-based, NetKet-inspired neural quantum state
framework centered on 2D TFIM and 2D J1-J2. The public API should stay close to
the workflow already expressed in the notebook, while the internals remain
project-owned and simple enough to validate against exact diagonalization. RBM
comes first, FFNN and CNN follow under the same interface, and Renyi-2 via SWAP
is the main scalable entanglement observable for the neural quantum state
pipeline.
