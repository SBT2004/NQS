from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinHilbert:
    """Spin-1/2 Hilbert space with states encoded as {0, 1} arrays."""

    N: int

    def __post_init__(self) -> None:
        if not isinstance(self.N, int):
            raise TypeError("N must be an integer.")
        if self.N <= 0:
            raise ValueError("N must be positive.")
        if self.N > 128:
            raise ValueError("Stage 1 supports at most 128 spins.")

    @property
    def size(self) -> int:
        return self.N

    @property
    def n_states(self) -> int:
        return 1 << self.N

    def validate_state(self, state: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(state, dtype=np.uint8)
        if arr.shape != (self.N,):
            raise ValueError(f"Expected shape ({self.N},), got {arr.shape}.")
        if np.any((arr != 0) & (arr != 1)):
            raise ValueError("Spin states must only contain 0 or 1.")
        return arr

    def state_to_index(self, state: np.ndarray | list[int] | tuple[int, ...]) -> int:
        arr = self.validate_state(state)
        return int(sum(int(bit) << i for i, bit in enumerate(arr.tolist())))

    def index_to_state(self, index: int) -> np.ndarray:
        if not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if index < 0 or index >= self.n_states:
            raise ValueError(f"index must lie in [0, {self.n_states}).")
        return np.array([(index >> i) & 1 for i in range(self.N)], dtype=np.uint8)

    def states_to_bits(self, states: np.ndarray | list[list[int]]) -> np.ndarray:
        arr = np.asarray(states, dtype=np.uint8)
        if arr.ndim == 1:
            return np.array([self.state_to_index(arr)], dtype=object)
        if arr.ndim != 2 or arr.shape[1] != self.N:
            raise ValueError(f"Expected shape (batch, {self.N}), got {arr.shape}.")
        return np.array([self.state_to_index(row) for row in arr], dtype=object)

    def bits_to_states(self, bits: int | list[int] | np.ndarray, N: int | None = None) -> np.ndarray:
        if N is not None and N != self.N:
            raise ValueError("bits_to_states only supports the Hilbert-space size of this instance.")
        if isinstance(bits, (int, np.integer)):
            return self.index_to_state(int(bits))

        bit_list = np.asarray(bits, dtype=object).reshape(-1)
        return np.stack([self.index_to_state(int(bit)) for bit in bit_list], axis=0)

    def all_states(self) -> np.ndarray:
        states = np.zeros((self.n_states, self.N), dtype=np.uint8)
        for index in range(self.n_states):
            states[index] = self.index_to_state(index)
        return states

    def states_to_pm1(self, states: np.ndarray | list[list[int]]) -> np.ndarray:
        arr = np.asarray(states, dtype=np.int8)
        if np.any((arr != 0) & (arr != 1)):
            raise ValueError("Spin states must only contain 0 or 1.")
        return 2 * arr - 1
