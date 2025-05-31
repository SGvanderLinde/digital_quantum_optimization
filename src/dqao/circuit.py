"""This module builds quatum circuits for quantum adiabetic optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, MappingProxyType, SupportsFloat, SupportsInt

import numpy as np
import pennylane as qml

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from pennylane.measurements import CountsMP

    from dqao.qubo import QUBO


class CounterDiabaticQuantumOptimizer:
    metadata = MappingProxyType({"counter_diabatic_layer_types": {"y", "yzy", None}})

    def __init__(
        self,
        n_layers: SupportsInt,
        device: qml.Device,
        counter_diabatic_layer: str | None = "y",
        end_time: SupportsFloat = 1.0,
    ) -> None:
        r"""Initialize the ``CounterDiabaticQuantumOptimizer``.

        Args:
            n_layers: integer value representing the number of layers. The number of
                layers is equal to the number of Trotter steps.
            device: PennyLane device to use.
            counter_diabatic_layer: Choose  ``"Y"`` or ``YZY``  for the respectice
                counter diabatic layers as described in [1]. Choose ``None`` when no
                counter diabatic layer is desired. Default is "Y".
            end_time: end time of the annealing schedule. This effects the derivative
                of the annealing schedule and thus the strength of the counterdiabatic
                term. Larger values of `end_time` reduce the strength of the
                counterdiabatic term.


        [1] Hegade, N. N., Chen, X., & Solano, E. (2022). Digitized counterdiabatic
        quantum optimization. Physical Review Research, 4(4), L042030.

        """
        self._n_layers = int(n_layers)
        self._device = device
        self.end_time = float(end_time)

        if counter_diabatic_layer is None:
            cd_layer = None
        else:
            cd_layer = counter_diabatic_layer.strip().lower()
        supported_cd_layers = self.metadata["counter_diabatic_layer_types"]
        if cd_layer not in supported_cd_layers:
            error_msg = (
                f"unknown 'counter_diabatic_layer', choose from {supported_cd_layers}"
            )
            raise ValueError(error_msg)

        self._cd_layer = cd_layer

    @property
    def end_time(self) -> float:
        """End time of the annealing schedule."""
        return self._end_time

    @end_time.setter
    def end_time(self, end_time: SupportsFloat) -> None:
        try:
            end_time = float(end_time)
        except TypeError as e:
            error_msg = (
                "'end_time' must be of type 'SupportsFloat', but was of type "
                f"{type(end_time)}"
            )
            raise TypeError(error_msg) from e

        if end_time <= 0:
            error_msg = f"'end_time' must be strictly positive, but was {end_time}"
            raise ValueError(error_msg)

        self._end_time = end_time

    def sample_qubo(self, qubo: QUBO) -> dict[str, int]:
        """Sample a ``QUBO`` problem.

        The ``QUBO`` problem is transformed to the corresponding Lenz-Ising model. This
        Lenz-Ising model is encoded in the cost and counterdiabatic layers.

        Args:
            qubo: ``QUBO`` problem to sample.

        Returns:
            Dictionary with samples. Each key is a bitstring and each correspending
            value is the numner of times that bitstring was measured.

        """
        return self.sample_lenz_ising(*qubo.to_lenz_ising())

    def sample_lenz_ising(
        self,
        interactions: ArrayLike,
        bias_terms: ArrayLike,
        offset: SupportsFloat = 0,  # noqa: ARG002
    ) -> dict[str, int]:
        """Sample a Lenz-Ising problem.

        The Lenz-Ising model is encoded in the cost and counterdiabatic layers.

        Args:
            interactions: interaction terms of the Lenz-Ising problem. These are the
                quadratic terms of the problem.
            bias_terms: bias terms of the Lenz-Ising problem. These are the linear
                terms of the problem.
            offset: constant offset.

        Returns:
            Dictionary with samples. Each key is a bitstring and each correspending
            value is the numner of times that bitstring was measured.

        """
        circuit = self.build_qnode(interactions, bias_terms)
        return circuit()

    def build_qnode(self, interactions: ArrayLike, bias_terms: ArrayLike) -> qml.QNode:
        interactions = np.asarray(interactions, dtype=np.float64)
        bias_terms = np.asarray(bias_terms, dtype=np.float64)

        norm_constant = max(interactions.max(), bias_terms.max())
        interactions /= norm_constant
        bias_terms /= norm_constant

        circuit = QFunc(
            bias_terms,
            interactions,
            self._n_layers,
            self._cd_layer,
            self._end_time,
        )

        return qml.QNode(circuit, self._device)


class QFunc:
    def __init__(
        self,
        bias_terms: ArrayLike,
        interactions: ArrayLike,
        n_layers: SupportsInt,
        cd_layer: str | None,
        end_time: SupportsFloat,
    ) -> None:
        self._bias_terms = np.asarray(bias_terms, dtype=np.float64)
        self._interactions = np.asarray(interactions, dtype=np.float64)
        self._n_layers = int(n_layers)
        self._cd_layer = cd_layer
        self._end_time = float(end_time)

    @property
    def n_qubits(self) -> int:
        return len(self._bias_terms)

    def __call__(self) -> CountsMP:
        timesteps = np.linspace(0, 1, self._n_layers + 2)[1:-1]
        schedule = np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * timesteps) ** 2) ** 2

        schedule_grad = (
            np.pi**2
            * np.sin(timesteps * np.pi)
            * np.sin(np.pi * np.sin(0.5 * np.pi * timesteps) ** 2)
            / (4 * self._end_time)
        )

        cost_ham_layer = CostLayer(self._bias_terms, self._interactions)
        initial_ham_layer = InitialLayer(self.n_qubits)

        if self._cd_layer == "y":
            cd_ham_layer = CDLayerY(self._bias_terms, self._interactions)
        elif self._cd_layer == "yzy":
            cd_ham_layer = CDLayerYZY(self._bias_terms, self._interactions)
        else:
            cd_ham_layer = CDLayerNull()

        initial_ham_layer.prep()

        for ts, strength in enumerate(schedule):
            cost_ham_layer(strength)
            initial_ham_layer(strength)

            # Counterdiabetic term
            cd_ham_layer(strength, schedule_grad[ts])

        return qml.counts()


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
