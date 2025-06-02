"""Moduke containing layers for digital quantum optimization circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsInt

import numpy as np
import pennylane as qml

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class CostLayer:
    def __init__(self, bias_terms: ArrayLike, interactions: ArrayLike) -> None:
        """Init the CostLayer.

        Args:
            bias_terms: 1D-ArrayLike representing the bias terms of a Lenz-Ising
                problem. These are the linear terms of the problem.
            interactions: 2D-ArrayLike representing the interaction terms of the
                Lenz-Ising problem. These are the quadratic terms of the problem.
        """
        self._bias_terms = np.asarray(bias_terms, dtype=np.float64)
        self._interactions = np.asarray(interactions, dtype=np.float64)

    @property
    def n_qubits(self) -> int:
        """Number of qubits used by the circuit."""
        return len(self._bias_terms)

    def __call__(self, angle: float) -> None:
        r"""Apply the cost layer.

        The cost layer is defined as

        .. math::

            \theta \\left(
                \\sum_{i=1}^N h_i Z_i + \\sum_{i=1}^N\\sum_{j=i+1}^NJ_{ij}Z_iZ_j
            \right).

        **Notation**

        * $h_i$: linear bias term $i$ of the Lenz-Ising model.
        * $J_{ij}$: quadratic interaction term acting on qubits $i$ and $j$.
        * $N$: number of spins in the Lenz-Ising model.
        * $Z_i$: Pauli-Z operator acting on qubit $i$.
        * $\theta$: angle provided to the call of this method.

        Args:
            angle: $\theta$ to use when applying the cost layer.
        """
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
        """Initialze the InitialLayer.

        Args:
            n_qubits: number of qubits.
        """
        self._n_qubits = int(n_qubits)

    def prep(self) -> None:
        """State preperation for this InitialLayer.

        For the adiabatic/counterdiabatic quantum optimization protocol, the qubits
        need to be prepared in the ground state of the initial Hamiltonian. For this
        initial Hamiltonian the ground state is a equal superposition, i.e.,

        .. math::

            \\sum_{i=1}^N |+\rangle.

        **Notation**

        * $N$: number of spins in the Lenz-Ising model.
        """
        for i in range(self._n_qubits):
            qml.Hadamard(wires=i)

    def __call__(self, angle: float) -> None:
        r"""Apply the layer of an initial Hamiltonian.
        The initial Hamiltonian refers to the initial Hamiltonian in the
        adiabatic/counterdiabatic quantum optimization protocol.

        The initial layer is defined as

        .. math::

            \theta \sum_{i=1}^N X_i.

        **Notation**

        * $N$: number of spins in the Lenz-Ising model.
        * $X_i$: Pauli-X operator acting on qubit $i$.
        * $\theta$: angle provided to the call of this method.

        Args:
            angle: $\theta$ to use when applying the initial layer.
        """
        for i in range(self._n_qubits):
            qml.RX(-2 * (1 - angle), wires=i)


class CDLayerY:
    def __init__(self, bias_terms: ArrayLike, interactions: ArrayLike) -> None:
        """Init the CDLayerY as described in [1].

        Args:
            bias_terms: 1D-ArrayLike representing the bias terms of a Lenz-Ising
                problem. These are the linear terms of the problem.
            interactions: 2D-ArrayLike representing the interaction terms of the
                Lenz-Ising problem. These are the quadratic terms of the problem.

        [1] Hegade, N. N., Chen, X., & Solano, E. (2022). Digitized counterdiabatic
        quantum optimization. Physical Review Research, 4(4), L042030.
        """
        self._bias_terms = np.asarray(bias_terms, dtype=np.float64)
        self._interactions = np.asarray(interactions, dtype=np.float64)

    @property
    def n_qubits(self) -> int:
        """Number of qubits used by the circuit."""
        return len(self._bias_terms)

    def __call__(self, strength: float, strength_grad: float) -> None:
        r"""Apply the Y counterdiabatic layer.

        The Y counterdiabatic layer is defined as

        .. math::

            \dot{\lambda(t)}\frac{h_i}{2}
            \sum_{i=1}^NY_i
            \left[
            (\lambda(t)-1)^2
            + \lambda(t)^2\left(h_i^2 + \sum_{j=1}^N(J_{ij} + J_{ji})\right)
            \right]

        **Notation**

        * $h_i$: linear bias term $i$ of the Lenz-Ising model.
        * $J_{ij}$: quadratic interaction term acting on qubits $i$ and $j$.
        * $N$: number of spins in the Lenz-Ising model.
        * $Y_i$: Pauli-Y operator acting on qubit $i$.
        * $\lambda$: strength schedule.

        Args:
            strength: strength of the schedule at timestep $t$.
            strength_grad: gradient strength of the schedule at timestep $t$.
        """
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
        """Do nothing."""
        return


class CDLayerYZY:
    def __init__(self, bias_terms: ArrayLike, interactions: ArrayLike) -> None:
        """Init the CDLayerYZY as described in [1].

        Args:
            bias_terms: 1D-ArrayLike representing the bias terms of a Lenz-Ising
                problem. These are the linear terms of the problem.
            interactions: 2D-ArrayLike representing the interaction terms of the
                Lenz-Ising problem. These are the quadratic terms of the problem.

        [1] Hegade, N. N., Chen, X., & Solano, E. (2022). Digitized counterdiabatic
        quantum optimization. Physical Review Research, 4(4), L042030.
        """
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
        """Number of qubits used by the circuit."""
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
