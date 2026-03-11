import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nqs import SpinHilbert


class SpinHilbertTests(unittest.TestCase):
    def test_index_roundtrip(self) -> None:
        hilbert = SpinHilbert(5)
        for index in [0, 1, 7, 16, 31]:
            state = hilbert.index_to_state(index)
            self.assertEqual(hilbert.state_to_index(state), index)

    def test_bits_roundtrip_batch(self) -> None:
        hilbert = SpinHilbert(4)
        states = np.array([[0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 1, 0]], dtype=np.uint8)
        bits = hilbert.states_to_bits(states)
        recovered = hilbert.bits_to_states(bits)
        np.testing.assert_array_equal(recovered, states)

    def test_all_states_order(self) -> None:
        hilbert = SpinHilbert(3)
        all_states = hilbert.all_states()
        self.assertEqual(all_states.shape, (8, 3))
        np.testing.assert_array_equal(all_states[5], np.array([1, 0, 1], dtype=np.uint8))

    def test_invalid_values_are_rejected(self) -> None:
        hilbert = SpinHilbert(3)
        with self.assertRaises(ValueError):
            hilbert.validate_state([0, 2, 1])

    def test_oversized_hilbert_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpinHilbert(129)


if __name__ == "__main__":
    unittest.main()
