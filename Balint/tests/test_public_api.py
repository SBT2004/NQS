import unittest

import nqs


class PublicApiTests(unittest.TestCase):
    def test_notebook_style_module_access_smoke(self) -> None:
        self.assertIsNotNone(nqs.graph.SquareLattice)
        self.assertIsNotNone(nqs.hilbert.SpinHilbert)
        self.assertIsNotNone(nqs.operator.tfim)
        self.assertIsNotNone(nqs.models.RBM)
        self.assertIsNotNone(nqs.sampler.NetKetSampler)
        self.assertIsNotNone(nqs.vqs.VariationalState)
        self.assertIsNotNone(nqs.optimizer.Adam)
        self.assertIsNotNone(nqs.driver.VMC)
        self.assertIsNotNone(nqs.observables.entropy_callback)


if __name__ == "__main__":
    unittest.main()
