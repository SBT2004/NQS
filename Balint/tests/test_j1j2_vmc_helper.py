from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, cast

import netket as nk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.j1j2_vmc_helper import (  # noqa: E402
    build_j1j2_operators,
    format_runtime_change_report,
    generate_j1j2_runtime_report,
    run_j1j2_rbm_demo,
)


class J1J2HelperTests(unittest.TestCase):
    def test_build_j1j2_operators_returns_both_operator_paths(self) -> None:
        hilbert, graph, native_operator, project_operator = build_j1j2_operators(Lx=2, Ly=2, pbc=False)

        self.assertEqual(hilbert.size, 4)
        self.assertEqual(graph.n_nodes, 4)
        self.assertIsInstance(native_operator, nk.operator.LocalOperator)
        self.assertIsInstance(project_operator, nk.operator.LocalOperator)

    def test_speed_mode_runs_only_project_operator_and_records_iteration_timing(self) -> None:
        result = run_j1j2_rbm_demo(
            Lx=2,
            Ly=2,
            J1=1.0,
            J2=0.5,
            pbc=False,
            n_iter=2,
            n_samples=16,
            n_discard_per_chain=4,
            n_chains=4,
            eval_samples=32,
            eval_repeats=1,
            mode="speed",
        )

        results = cast(list[dict[str, Any]], result["results"])
        self.assertEqual(result["mode"], "speed")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "project_operator")
        self.assertIsNone(result["energy_gap"])
        self.assertEqual(len(cast(list[dict[str, Any]], results[0]["iteration_timing"])), 2)

    def test_comparison_mode_keeps_energy_gap_within_tolerance(self) -> None:
        result = run_j1j2_rbm_demo(
            Lx=2,
            Ly=2,
            J1=1.0,
            J2=0.5,
            pbc=False,
            n_iter=8,
            n_samples=64,
            n_discard_per_chain=8,
            n_chains=8,
            eval_samples=256,
            eval_repeats=1,
            mode="comparison",
        )

        self.assertLess(float(cast(dict[str, Any], result)["energy_gap"]), 0.1)

    def test_runtime_report_includes_size_and_iteration_tables(self) -> None:
        report = generate_j1j2_runtime_report(
            sizes=((2, 2),),
            comparison_checkpoints=frozenset({(2, 2)}),
            n_iter=2,
            n_samples=16,
            n_discard_per_chain=4,
            n_chains=4,
            eval_samples=32,
            eval_repeats=1,
            warm_runs=1,
        )
        report_text = format_runtime_change_report(report)

        self.assertEqual(len(cast(list[dict[str, Any]], report["size_scaling"])), 1)
        self.assertIn("Size scaling", report_text)
        self.assertIn("Iteration timing", report_text)
        self.assertIn("2x2", report_text)


if __name__ == "__main__":
    unittest.main()
