from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pennylane as qml
from dqao import QUBO, optimize_cdqo
from dqao import CounterDiabaticQuantumOptimizer as CDQOptimizer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def run(matrix: ArrayLike, solution: str) -> None:
    qubo = QUBO(matrix)
    device = qml.device("default.qubit", shots=10000)

    n_layers_iter = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30]
    results = {"No CD": [], "Y": [], "YZY": []}

    for n_layers in n_layers_iter:
        optimizer = CDQOptimizer(n_layers, device, None)
        counts_none = optimizer.sample_qubo(qubo)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer = CDQOptimizer(n_layers, device, "y")
            counts_y = optimize_cdqo(optimizer, qubo)

            optimizer = CDQOptimizer(n_layers, device, "yzy")
            counts_yzy = optimize_cdqo(optimizer, qubo)

        total_shots = device.shots.total_shots
        succ_none = counts_none.get(solution, 0) / total_shots
        succ_y = counts_y.get(solution, 0) / total_shots
        succ_yzy = counts_yzy.get(solution, 0) / total_shots

        _pprint_iter(n_layers, succ_none, succ_y, succ_yzy, optimizer.end_time)

        results["No CD"].append(succ_none)
        results["Y"].append(succ_y)
        results["YZY"].append(succ_yzy)

    filepath = Path(__file__).parent / "figures" / f"plot{len(qubo)}.jpg"
    _plot_results(n_layers_iter, results, filepath)


def _pprint_iter(
    n_layers: int,
    succ_none: float,
    succ_y: float,
    succ_yzy: float,
    end_time: float,
) -> None:
    print(
        f"{n_layers:>5} layers: {succ_none:.2%} {succ_y:.2%} {succ_yzy:.2%} "
        f"({end_time:.1f})",
    )


def _plot_results(
    x_axis: list[int],
    results: dict[str, list[float]],
    filepath: Path | str,
) -> None:
    fig, ax = plt.subplots()
    for name, y_axis in results.items():
        ax.plot(x_axis, y_axis, label=name)

    ax.legend()
    ax.set(xlabel="n_layers", ylabel="succes ratio")
    fig.savefig(filepath)


if __name__ == "__main__":
    MATRIX = [[1, 2, 3], [0, -50, 4], [0, 0, 6]]
    run(MATRIX, "010")

    MATRIX = [[1, 2, 3, 4], [0, -99, 5, 6], [0, 0, 7, 8], [0, 0, 0, 9]]
    run(MATRIX, "0100")

    MATRIX = [
        [1, 2, 3, 4, 5],
        [0, -99, 6, 7, 8],
        [0, 0, 9, 10, 11],
        [0, 0, 0, 12, 13],
        [0, 0, 0, 0, 14],
    ]
    run(MATRIX, "01000")
