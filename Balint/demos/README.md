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

## Reproducibility And Reruns

- All retained notebooks call `bootstrap_notebook(enable_x64=True)`, so the report runs assume JAX `x64` is enabled inside the notebook session.
- The exercise notebooks rewrite their exported CSV/HTML/PNG artifacts under `demos/report_outputs/exercise_1/`, `demos/report_outputs/exercise_2/`, and `demos/report_outputs/exercise_3/`.
- `demos/netket_benchmark.ipynb` is appendix-style validation content. Its rendered outputs are stored inline in the notebook rather than in a separate export directory.
- The notebooks are intended to be rerun top-to-bottom with `Run All` from Jupyter, or from the repo root with commands such as `jupyter nbconvert --to notebook --execute --inplace demos/exercise_1.ipynb`.
- Seed schedules, sample counts, backend assumptions, and output locations are repeated inside each notebook in a dedicated report-setup section so the report surface can be inspected without reconstructing hidden context from helper code.
