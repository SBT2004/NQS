import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.final_exam_helper import build_output_manifest, plot_training_history, save_report_figure, save_report_table  # noqa: E402


class FinalExamHelperTests(unittest.TestCase):
    def test_save_report_artifacts_and_manifest(self) -> None:
        summary_table = pd.DataFrame(
            [
                {"model": "RBM", "half_partition_renyi2": 0.5, "parameter_count": 12},
                {"model": "FFNN", "half_partition_renyi2": 0.7, "parameter_count": 20},
            ]
        )
        history_table = pd.DataFrame(
            [
                {"step": 0, "renyi2_entropy": 0.2, "sweep_label": "critical"},
                {"step": 1, "renyi2_entropy": 0.4, "sweep_label": "critical"},
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            table_paths = save_report_table(summary_table, "architecture_summary", output_dir=temp_dir)
            figure = plot_training_history(history_table, "renyi2_entropy")
            figure_path = save_report_figure(figure, "training_history", output_dir=temp_dir)
            plt.close(figure)

            self.assertTrue(table_paths["csv"].exists())
            self.assertTrue(table_paths["html"].exists())
            self.assertTrue(figure_path.exists())

            manifest = build_output_manifest(
                [
                    {"section": "training", "name": "history", "path": str(figure_path)},
                    {"section": "architecture", "name": "summary", "path": str(table_paths["csv"])},
                ]
            )
            self.assertEqual(manifest["section"].tolist(), ["architecture", "training"])


if __name__ == "__main__":
    unittest.main()
