# Perfetto Trace Summary

- Run dir: `C:\Users\balin\PycharmProjects\NQS\Balint\demos\profiling_results\codex_visibility_test`
- Dominant phase: `compile_heavy`
- Total trace size (bytes): `9213455`
- Event count: `1000042`
- Viewer hint: Perfetto UI is for human exploration. Codex analysis should use summary.json and summary.md from the profiling run directory.

## Run Metadata

- run_name: `codex_visibility_test`
- model_name: `CNN`
- length: `5`
- transverse_field: `1.0`
- fast_profile: `True`

## Top Events

- `$performance_profiling_helper.py:304 _run_profile_model` (uncategorized) - 6167.558 ms
- `$ising1d_ed_vs_vmc_helper.py:49 run_model_demo` (uncategorized) - 6159.079 ms
- `$driver.py:89 run` (uncategorized) - 2669.197 ms
- `$driver.py:63 step` (uncategorized) - 2557.640 ms
- `$vqs.py:55 expect_and_grad` (uncategorized) - 2549.376 ms
- `$timing.py:288 timed_function` (uncategorized) - 2549.336 ms
- `$state.py:656 expect_and_grad` (uncategorized) - 2549.311 ms
- `$_function.py:392 __call__` (uncategorized) - 2549.304 ms
- `$expect_grad.py:41 expect_and_grad_default_formula` (uncategorized) - 2548.990 ms
- `$_function.py:392 __call__` (uncategorized) - 2548.799 ms
