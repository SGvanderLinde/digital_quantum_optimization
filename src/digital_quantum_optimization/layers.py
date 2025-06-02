from __future__ import annotations

from typing import TYPE_CHECKING, SupportsInt

import numpy as np
import pennylane as qml

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class CostLayer:
    def __init__(self, bias_terms: ArrayLike, interactions: ArrayLike) -> None:
        self._bias_terms = np.asarray(bias_terms, dtype=np.float64)
        self._interactions = np.asarray(interactions, dtype=np.float64)

    @property
    def n_qubits(self) -> int:
        return len(self._bias_terms)

    def __call__(self, angle: float) -> None:
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                interaction = self._interactions[i, j]
                if interaction == 0:
                    continue
                qml.MultiRZ(2 * angle * interaction, wires=[i, j])

        for i, bias in enumerate(self._bias_terms):
            if bias:
                qml.RZ(2 * angle * bias, wires=i)


class InitialLayer:
    def __init__(self, n_qubits: SupportsInt) -> None:
        self._n_qubits = int(n_qubits)

    def prep(self) -> None:
        for i in range(self._n_qubits):
            qml.Hadamard(wires=i)

    def __call__(self, angle: float) -> None:
        for i in range(self._n_qubits):
            qml.RX(-2 * (1 - angle), wires=i)


class CDLayerY:
    def __init__(self, bias_terms: ArrayLike, interactions: ArrayLike) -> None:
        self._bias_terms = np.asarray(bias_terms, dtype=np.float64)
        self._interactions = np.asarray(interactions, dtype=np.float64)

    @property
    def n_qubits(self) -> int:
        return len(self._bias_terms)

    def __call__(self, strength: float, strength_grad: float) -> None:
        for i, bias in enumerate(self._bias_terms):
            angle = bias**2
            angle += np.sum(self._interactions[i] ** 2)
            angle += np.sum(self._interactions[:, i] ** 2)
            angle *= strength**2
            angle += (strength - 1) ** 2
            angle *= bias * strength_grad

        qml.RY(angle, wires=i)


class CDLayerNull:
    def __call__(self, strength: float, strength_grad: float) -> None:  # noqa: ARG002
        return None


class CDLayerYZY:
    def __init__(self, bias_terms: ArrayLike, interactions: ArrayLike) -> None:
        self._bias_terms = np.asarray(bias_terms, dtype=np.float64)
        self._interactions = np.asarray(interactions, dtype=np.float64)

        const1, const2, const3 = self._compute_schedule_consts(
            self._bias_terms,
            self._interactions,
        )
        self._const1 = const1
        self._const2 = const2
        self._const3 = const3

        self._bias_terms = bias_terms
        self._interactions = interactions

    @property
    def n_qubits(self) -> int:
        return len(self._bias_terms)

    @staticmethod
    def _compute_schedule_consts(
        bias_terms: NDArray[np.float64],
        interactions: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        n_qubits = len(bias_terms)
        const1 = np.sum(bias_terms**2) + 8 * np.sum(interactions**2)
        const2 = const1 + np.sum(bias_terms**4) + 2 * np.sum(interactions**4)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                const2 += 6 * bias_terms[i] ** 2 * interactions[i, j] ** 2
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for k in range(n_qubits):
                    for l in range(k + 1, n_qubits):
                        if i != k and j != l:
                            continue
                        const2 += 6 * interactions[i, j] ** 2 * interactions[k, l] ** 2

        const3 = -0.25 * (np.sum(bias_terms**2) + 2 * np.sum(interactions**2))
        return const1, const2, const3

    def __call__(self, strength: float, strength_grad: float) -> None:
        R_ts = (1 - 2 * strength) * self._const1 + strength**2 * self._const2
        alpha = self._const3 / R_ts

        angle = -2 * strength_grad * alpha

        for i, bias in enumerate(self._bias_terms):
            if bias == 0:
                continue
            qml.RY(2 * angle * bias, wires=i)

        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                interaction = self._interactions[i, j]
                if interaction == 0:
                    continue
                qml.PauliRot(2 * angle * interaction, "YZ", wires=[i, j])
                qml.PauliRot(2 * angle * interaction, "ZY", wires=[i, j])
