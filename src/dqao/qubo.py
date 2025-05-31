"""This module contains the QUBO object and related helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class QUBO:
    """Representstion of a Quadratic Unconstrained Binary Optimiazation (QUBO) problem.

    The QUBO problem has a strong relation with the Lenz-Ising model. The corresponding
    Hamiltonian of the Lenz-Ising  problem can be easily expressed by a quantum circuit.
    """

    def __init__(self, matrix: ArrayLike, offset: SupportsFloat = 0.0) -> None:
        """Init of the ``QUBO`` object.

        Args:
            matrix: 2-D square ArrayLike representstion of the QUBO matrix.
            offset: Optional offset of the problem. Default value is 0.0.
        """
        self._matrix = np.array(matrix, dtype=np.float64)
        self._offset = float(offset)

    def to_lenz_ising(self) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        r"""Create the corresponding Lenz-Ising problem.

        This method uses to mapping $x_i \to (1-s_i)/2$.

        Returns:
            Tuple containing the interactions, bias terms and ofsett of the Lenz-Ising
            problem.
        """
        interactions = 0.25 * np.triu(self._matrix + self._matrix.T, k=1)
        bias_terms = -0.25 * (self._matrix.sum(axis=0) + self._matrix.sum(axis=1))
        offset = self._offset + 0.25 * (
            self._matrix.sum() + np.diag(self._matrix).sum()
        )

        return interactions, bias_terms, float(offset)

    def evaluate(self, bits: ArrayLike) -> float:
        """Evaluate the QUBO problem for the provided `bits`.

        Args:
            bits: 1-D ArrayLike to evalutate the QUBO provlem for.

        Returns:
            Corresponding value of the provided `bits`.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        value = bits @ self._matrix @ bits + self._offset
        return float(value)

    def brute_force(self) -> tuple[NDArray[np.uint8], float]:
        """Find the optimal bits-value pair of the QUBO problem using brute force.

        This is a naive brute force implementation with a computational complexity of $O(2^n n^2)$.

        Returns:
            Tuple with the optimal bits and optimal value of the QUBO problem.
        """
        best_value = self._offset
        best_bits = np.zeros(len(self), dtype=np.uint8)
        for i in range(1, len(self) ** 2 - 1):
            bit_string = np.binary_repr(i, width=len(self))
            bits = np.fromiter(bit_string, np.uint8, len(self))
            value = self.evaluate(bits)
            if value < best_value:
                best_value = value
                best_bits = bits

        return best_bits, best_value

    def __len__(self) -> int:
        """Number of variables of the QUBO problem."""
        return int(self._matrix.shape[0])

    def __eq__(self, other: Any) -> bool:
        """Returns true if self is equal to other.

        Two QUBO object are equal if their offsets are equal and their matrices are
        equal in symmetric form.

        Args:
            other: object to compare against.

        Returns:
            Boolean value stating wether the two QUBO objects are equal.
        """
        if not isinstance(other, QUBO):
            return False

        if self._offset != other._offset:
            return False

        comp_matrix1 = self._matrix + self._matrix.T
        comp_matrix2 = other._matrix + other._matrix.T

        return np.array_equal(comp_matrix1, comp_matrix2)
