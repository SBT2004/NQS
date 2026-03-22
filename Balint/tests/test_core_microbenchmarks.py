from __future__ import annotations

import sys
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import TypeVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqs.exact_diag import solve_sparse_ground_state, sparse_operator_matrix
from nqs.graph import Chain1D
from nqs.hilbert import SpinHilbert
from nqs.operator import Operator, collect_terms, sx_term, szsz_term
from nqs.workflows import run_incremental_exercise_1_ed_benchmark


@dataclass(frozen=True)
class MicrobenchmarkResult:
    name: str
    setup_ms: float
    first_run_ms: float
    steady_state_mean_ms: float
    steady_state_min_ms: float
    repeats: int


TContext = TypeVar("TContext")
TResult = TypeVar("TResult")


def _elapsed_ms(fn: Callable[[], TResult]) -> tuple[TResult, float]:
    start = time.perf_counter_ns()
    result = fn()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return result, elapsed_ms


def _measure_microbenchmark(
    name: str,
    setup_fn: Callable[[], TContext],
    run_fn: Callable[[TContext], TResult],
    validate_fn: Callable[[TContext, TResult], None],
    *,
    repeats: int,
) -> MicrobenchmarkResult:
    context, setup_ms = _elapsed_ms(setup_fn)
    first_result, first_run_ms = _elapsed_ms(lambda: run_fn(context))
    validate_fn(context, first_result)
    steady_state_runs: list[float] = []
    for _ in range(repeats):
        _, run_ms = _elapsed_ms(lambda: run_fn(context))
        steady_state_runs.append(run_ms)
    return MicrobenchmarkResult(
        name=name,
        setup_ms=setup_ms,
        first_run_ms=first_run_ms,
        steady_state_mean_ms=mean(steady_state_runs),
        steady_state_min_ms=min(steady_state_runs),
        repeats=repeats,
    )


def _build_chain_tfim_operator(n_spins: int, *, field_strength: float) -> Operator:
    hilbert = SpinHilbert(n_spins)
    graph = Chain1D(length=n_spins, pbc=False)
    interaction_terms = [szsz_term(edge.i, edge.j, coefficient=-1.0) for edge in graph.iter_edges("J", n=1)]
    field_terms = [sx_term(site, coefficient=-field_strength) for site in range(n_spins)]
    return Operator(hilbert, collect_terms(interaction_terms, field_terms))


def _validate_hilbert_roundtrip(
    context: tuple[SpinHilbert, np.ndarray],
    result: np.ndarray,
) -> None:
    np.testing.assert_array_equal(result, context[1])


def _validate_operator_connected_elements(
    context: tuple[Operator, int],
    result: list[tuple[int, complex]],
) -> None:
    operator, state_bits = context
    state = operator.hilbert.index_to_state(state_bits)
    expected = [
        (operator.hilbert.state_to_index(connected_state), value)
        for connected_state, value in operator.connected_elements(state)
    ]
    assert result == expected


def _validate_sparse_operator_matrix(operator: Operator, matrix) -> None:
    dimension = operator.hilbert.n_states
    assert matrix.shape == (dimension, dimension)
    dense_matrix = matrix.toarray()
    np.testing.assert_allclose(dense_matrix, dense_matrix.conj().T)
    assert np.isclose(dense_matrix[0, 0], -(operator.hilbert.size - 1))
    assert np.isclose(dense_matrix[1, 0], -0.8)


def _validate_sparse_ground_state(_context, result: dict[str, np.ndarray | float]) -> None:
    assert np.isfinite(result["ground_energy"])
    assert np.isclose(np.linalg.norm(np.asarray(result["ground_state"], dtype=np.complex128)), 1.0)


def run_core_microbenchmarks() -> list[MicrobenchmarkResult]:
    hilbert_result = _measure_microbenchmark(
        "hilbert.state_bitmap_roundtrip",
        setup_fn=lambda: (
            SpinHilbert(12),
            np.random.default_rng(0).integers(0, 2, size=(128, 12), dtype=np.uint8),
        ),
        run_fn=lambda context: context[0].bits_to_states(context[0].states_to_bits(context[1])),
        validate_fn=_validate_hilbert_roundtrip,
        repeats=8,
    )
    operator_result = _measure_microbenchmark(
        "operator.connected_elements_bits",
        setup_fn=lambda: (
            _build_chain_tfim_operator(12, field_strength=0.8),
            0b1010_1100_0110,
        ),
        run_fn=lambda context: context[0].connected_elements_bits(context[1]),
        validate_fn=_validate_operator_connected_elements,
        repeats=16,
    )
    exact_diag_assembly_result = _measure_microbenchmark(
        "exact_diag.sparse_operator_matrix",
        setup_fn=lambda: _build_chain_tfim_operator(8, field_strength=0.8),
        run_fn=sparse_operator_matrix,
        validate_fn=_validate_sparse_operator_matrix,
        repeats=4,
    )
    exact_diag_solve_result = _measure_microbenchmark(
        "exact_diag.solve_sparse_ground_state",
        setup_fn=lambda: sparse_operator_matrix(_build_chain_tfim_operator(10, field_strength=0.8)),
        run_fn=solve_sparse_ground_state,
        validate_fn=_validate_sparse_ground_state,
        repeats=4,
    )
    return [hilbert_result, operator_result, exact_diag_assembly_result, exact_diag_solve_result]


def format_core_microbenchmark_table(results: list[MicrobenchmarkResult]) -> str:
    header = "Benchmark                        setup_ms  first_ms  steady_mean_ms  steady_min_ms  repeats"
    rows = [
        (
            f"{result.name:<32}"
            f"{result.setup_ms:>10.3f}"
            f"{result.first_run_ms:>10.3f}"
            f"{result.steady_state_mean_ms:>16.3f}"
            f"{result.steady_state_min_ms:>15.3f}"
            f"{result.repeats:>9}"
        )
        for result in results
    ]
    return "\n".join([header, *rows])


class CoreMicrobenchmarkTests(unittest.TestCase):
    def test_core_microbenchmarks_cover_hilbert_operator_and_exact_diag(self) -> None:
        results = run_core_microbenchmarks()

        self.assertEqual(
            [result.name for result in results],
            [
                "hilbert.state_bitmap_roundtrip",
                "operator.connected_elements_bits",
                "exact_diag.sparse_operator_matrix",
                "exact_diag.solve_sparse_ground_state",
            ],
        )
        self.assertTrue(all(result.setup_ms >= 0.0 for result in results))
        self.assertTrue(all(result.first_run_ms > 0.0 for result in results))
        self.assertTrue(all(result.steady_state_mean_ms > 0.0 for result in results))
        self.assertTrue(all(result.steady_state_min_ms > 0.0 for result in results))
        self.assertTrue(all(result.steady_state_min_ms <= result.steady_state_mean_ms for result in results))

    def test_core_microbenchmark_table_contains_all_paths(self) -> None:
        table = format_core_microbenchmark_table(run_core_microbenchmarks())

        self.assertIn("hilbert.state_bitmap_roundtrip", table)
        self.assertIn("operator.connected_elements_bits", table)
        self.assertIn("exact_diag.sparse_operator_matrix", table)
        self.assertIn("exact_diag.solve_sparse_ground_state", table)

    def test_incremental_exercise_1_ed_benchmark_tracks_runtime_and_completion(self) -> None:
        result = run_incremental_exercise_1_ed_benchmark([4, 6], h=1.0, pbc=False)

        self.assertEqual(result["length"].tolist(), [4, 6])
        self.assertTrue(result["completed"].all())
        self.assertTrue((result["runtime_seconds"] > 0.0).all())
        self.assertTrue((result["failure_type"] == "").all())
        self.assertTrue(np.isfinite(result["ground_energy"]).all())


if __name__ == "__main__":
    print(format_core_microbenchmark_table(run_core_microbenchmarks()))
