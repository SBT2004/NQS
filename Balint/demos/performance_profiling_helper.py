"""Helpers for profiling short VMC workloads in this repository.

This file is intentionally demo-oriented. It keeps the profiling setup in one
place so notebooks can focus on showing the workflow instead of repeating
boilerplate.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path
from typing import Any

import jax
import netket as nk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nqs import Adam, RBM, NetKetSampler, SpinHilbert, VMC, VariationalState  # noqa: E402


def build_short_vmc_problem(
    length: int = 5,
    transverse_field: float = 1.0,
    seed: int = 0,
    n_samples: int = 16,
    n_discard_per_chain: int = 2,
    n_chains: int = 4,
    learning_rate: float = 1e-2,
) -> tuple[VMC, dict[str, Any]]:
    """Create a small VMC problem that is fast enough for profiling demos.

    The setup is intentionally modest. Profilers add overhead, so a tiny system
    makes the tool easier to use while still exercising our VMC stack.
    """

    hilbert = SpinHilbert(length)
    model = RBM(alpha=1)
    params = model.init(jax.random.PRNGKey(seed), hilbert)
    sampler = NetKetSampler(
        hilbert=hilbert,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        seed=seed,
    )
    state = VariationalState(model=model, params=params, sampler=sampler)
    operator = nk.operator.IsingJax(  # pyright: ignore[reportCallIssue]
        hilbert=sampler.netket_hilbert,
        graph=nk.graph.Chain(length=length, pbc=False),
        h=transverse_field,
    )
    driver = VMC(
        operator=operator,
        variational_state=state,
        optimizer=Adam(learning_rate=learning_rate),
    )
    metadata = {
        "length": length,
        "transverse_field": transverse_field,
        "seed": seed,
        "n_samples": n_samples,
        "n_discard_per_chain": n_discard_per_chain,
        "n_chains": n_chains,
        "learning_rate": learning_rate,
    }
    return driver, metadata


def run_short_vmc(n_iter: int = 3) -> list[dict[str, object]]:
    """Run a small VMC job that is safe to use as a profiling target."""

    driver, _ = build_short_vmc_problem()
    return driver.run(n_iter)


def cprofile_vmc_run(
    n_iter: int = 3,
    *,
    sort_by: str = "tottime",
    top_n: int = 25,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Profile a short VMC run with cProfile and return a text summary.

    `cProfile` is best for Python-level orchestration costs such as driver
    loops, object rebuilding, and adapter logic. It is less informative for
    time spent deep inside JAX/XLA kernels.
    """

    profiler = cProfile.Profile()
    profiler.enable()
    history = run_short_vmc(n_iter=n_iter)
    profiler.disable()

    if output_path is not None:
        profiler.dump_stats(str(output_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_by)
    stats.print_stats(top_n)
    return {
        "history": history,
        "summary": stream.getvalue(),
        "stats_path": None if output_path is None else str(output_path),
    }


def trace_vmc_with_jax_profiler(
    log_dir: str | Path,
    *,
    n_iter: int = 3,
    create_perfetto_link: bool = False,
) -> list[dict[str, object]]:
    """Write a JAX profiler trace for a short VMC run.

    Use this when the bottleneck is likely in JAX execution rather than Python
    control flow. The trace can later be opened with TensorBoard or Perfetto.
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(
        str(log_dir),
        create_perfetto_link=create_perfetto_link,
    )
    try:
        history = run_short_vmc(n_iter=n_iter)
        jax.block_until_ready(jax.tree_util.tree_map(jax.numpy.asarray, history))
    finally:
        jax.profiler.stop_trace()
    return history


def scalene_command(
    target: str = "Balint/demos/performance_profiling_helper.py",
    *,
    n_iter: int = 3,
) -> str:
    """Return a ready-to-run Scalene command for this repo.

    Scalene is a CLI profiler, so the notebook usually just prints the command
    to run in a terminal.
    """

    return f"python -m scalene {target} -- --n-iter {n_iter}"


if __name__ == "__main__":
    iterations = 3
    if len(sys.argv) >= 3 and sys.argv[1] == "--n-iter":
        iterations = int(sys.argv[2])
    result = cprofile_vmc_run(n_iter=iterations, top_n=20)
    print(result["summary"])
