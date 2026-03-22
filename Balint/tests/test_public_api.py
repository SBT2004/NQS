import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nqs


class PublicApiTests(unittest.TestCase):
    def test_notebook_style_module_access_smoke(self) -> None:
        self.assertIsNotNone(nqs.graph.SquareLattice)
        self.assertIsNotNone(nqs.hilbert.SpinHilbert)
        self.assertIsNotNone(nqs.operator.tfim)
        self.assertIsNotNone(nqs.models.RBM)
        self.assertIsNotNone(nqs.sampler.MetropolisLocal)
        self.assertIsNotNone(nqs.vqs.VariationalState)
        self.assertIsNotNone(nqs.optimizer.Adam)
        self.assertIsNotNone(nqs.driver.VMC)
        self.assertIsNotNone(nqs.observables.entropy_callback)
        self.assertFalse(hasattr(nqs, "NetKetSampler"))

    def test_project_owned_runtime_import_and_vmc_setup_do_not_load_netket(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        code = f"""
import sys
sys.path.insert(0, r"{project_root / 'src'}")
from nqs.graph import Chain1D
from nqs.hilbert import SpinHilbert
from nqs.models import RBM
from nqs.operator import Operator, collect_terms, sx_term, szsz_term
from nqs.vmc_setup import build_vmc_driver

hilbert = SpinHilbert(4)
graph = Chain1D(length=4, pbc=False)
interaction_terms = [szsz_term(edge.i, edge.j, coefficient=-1.0) for edge in graph.iter_edges("J", n=1)]
field_terms = [sx_term(site, coefficient=-1.0) for site in range(hilbert.size)]
operator = Operator(hilbert, collect_terms(interaction_terms, field_terms))
build_vmc_driver(
    model=RBM(alpha=1),
    hilbert=hilbert,
    operator=operator,
    learning_rate=1e-2,
    seed=0,
    n_samples=16,
    n_discard_per_chain=2,
    n_chains=4,
)
print("netket" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
