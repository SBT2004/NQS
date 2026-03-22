import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nqs.workflows import (  # noqa: E402
    format_tfim_5x5_vmc_performance_report,
    run_tfim_5x5_vmc_performance_benchmark,
)


class TFIM5x5VMCPerformanceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = run_tfim_5x5_vmc_performance_benchmark(
            warmed_repeats=1,
            n_samples=128,
            model_eval_batch_size=64,
        )

    def test_benchmark_records_cold_and_warmed_stage_timings(self) -> None:
        timing_table = self.benchmark["timing_table"]

        self.assertEqual(
            timing_table["stage"].tolist(),
            [
                "model_log_psi",
                "sampler_draw",
                "local_energy",
                "local_energy_gradient",
                "vmc_step",
            ],
        )
        self.assertEqual(self.benchmark["benchmark_label"], "tfim_5x5_open_rbm_vmc")
        self.assertEqual(self.benchmark["system_label"], "tfim_5x5_open")
        self.assertEqual(self.benchmark["lattice_shape"], (5, 5))
        self.assertEqual(self.benchmark["model_name"], "RBM")
        self.assertEqual(self.benchmark["model_kwargs"], {"alpha": 2})
        self.assertFalse(self.benchmark["pbc"])
        self.assertEqual(self.benchmark["n_sites"], 25)
        self.assertEqual(self.benchmark["warmed_repeats"], 1)
        self.assertTrue((timing_table["cold_start_ms"] > 0.0).all())
        self.assertTrue((timing_table["warmed_mean_ms"] > 0.0).all())
        self.assertTrue((timing_table["warmed_min_ms"] > 0.0).all())
        self.assertTrue(np.isfinite(timing_table["cold_to_warm_ratio"]).all())
        self.assertTrue((timing_table["cold_to_warm_ratio"] > 0.0).all())
        self.assertTrue((timing_table["suggested_max_regression_ratio"] == 1.25).all())

    def test_benchmark_exposes_regression_gate_table_and_report(self) -> None:
        regression_gate_table = self.benchmark["regression_gate_table"]
        report = format_tfim_5x5_vmc_performance_report(self.benchmark)

        self.assertEqual(
            regression_gate_table["stage"].tolist(),
            [
                "model_log_psi",
                "sampler_draw",
                "local_energy",
                "local_energy_gradient",
                "vmc_step",
            ],
        )
        self.assertTrue((regression_gate_table["regression_metric"] == "warmed_mean_ms").all())
        self.assertTrue((regression_gate_table["suggested_max_regression_ratio"] == 1.25).all())
        self.assertTrue(regression_gate_table["gate_description"].str.contains("cold_start_ms").all())
        self.assertIn("Stage timings (ms):", report)
        self.assertIn("Regression gates:", report)
        self.assertIn("model_log_psi", report)
        self.assertIn("local_energy_gradient", report)
        self.assertIn("vmc_step", report)


if __name__ == "__main__":
    unittest.main()
