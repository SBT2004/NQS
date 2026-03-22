# Demo Layout

The retained user-facing notebook surface is:

- `demos/netket_benchmark.ipynb`: the comparison notebook for project-vs-reference runs.
- `demos/exercise_1.ipynb`: Exercise 1, keeping sub-exercises in exam order.
- `demos/exercise_2.ipynb`: Exercise 2, keeping sub-exercises in exam order. Its TFIM size-sweep section currently uses the square-lattice `(L, 1)` proxy rather than a dedicated 1D graph implementation.
- `demos/exercise_3.ipynb`: Exercise 3, keeping sub-exercises in exam order.

Shared helper boundary:

- `demos/notebook_bootstrap.py`: shared repo-root bootstrap and notebook-only JAX `x64` setup.
- `nqs.workflows`: shared experiment helpers used by the benchmark and exercise notebooks.
- `demos/exercise_report_helper.py`: shared plotting and report/export helpers used by the split exercise notebooks. The filename is legacy, but the helper remains part of the retained notebook surface.
- `demos/netket_reference.py`: explicit NetKet-only comparison helpers used by the benchmark notebook. This is the intended place for optional NetKet interop, including compatibility calls that rely on `Operator.to_netket()`.
