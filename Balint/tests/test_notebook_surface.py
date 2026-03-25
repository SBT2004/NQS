import sys
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.notebook_bootstrap import bootstrap_notebook, ensure_repo_root_on_path  # noqa: E402


class NotebookSurfaceTests(unittest.TestCase):
    def test_retained_top_level_notebooks_match_split_layout(self) -> None:
        notebook_paths = sorted(path.relative_to(PROJECT_ROOT / "demos").as_posix() for path in (PROJECT_ROOT / "demos").glob("*.ipynb"))
        nested_notebooks = list((PROJECT_ROOT / "demos").glob("*/*.ipynb"))

        self.assertEqual(
            notebook_paths,
            [
                "exercise_1.ipynb",
                "exercise_2.ipynb",
                "exercise_3.ipynb",
                "netket_benchmark.ipynb",
            ],
        )
        self.assertEqual(nested_notebooks, [])

    def test_notebook_bootstrap_finds_repo_root_from_demos_subdirectory(self) -> None:
        root = ensure_repo_root_on_path(PROJECT_ROOT / "demos" / "report_outputs")
        bootstrapped_root = bootstrap_notebook(PROJECT_ROOT / "demos")

        self.assertEqual(root, PROJECT_ROOT)
        self.assertEqual(bootstrapped_root, PROJECT_ROOT)
        import nqs  # noqa: PLC0415
        import nqs.workflows  # noqa: PLC0415

        self.assertTrue(hasattr(nqs, "graph"))
        self.assertTrue(hasattr(nqs.workflows, "run_vmc_experiment"))

    def test_exercise_3_notebook_does_not_depend_on_saved_exercise_2_csv_artifacts(self) -> None:
        notebook = json.loads((PROJECT_ROOT / "demos" / "exercise_3.ipynb").read_text(encoding="utf-8"))
        notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertNotIn("exercise_2_architecture_summary.csv", notebook_source)


if __name__ == "__main__":
    unittest.main()
