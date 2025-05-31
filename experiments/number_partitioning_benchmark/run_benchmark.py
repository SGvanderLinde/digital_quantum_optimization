from __future__ import annotations

import itertools
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml

from dqao import QUBO, benchmarking, optimize_cdqo_gridsearch
from dqao import CounterDiabaticQuantumOptimizer as CDQOptimizer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


NUMBERS = [1, 2, 3, 4, 6, 8]
N_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50]
NOISE = [0.001, 0.01, 0.1]
CD_TYPES = ["No CD", "Y", "YZY"]
N_RETRIES = list(range(10))

def run(numbers: ArrayLike) -> None:
    qubo = benchmarking.build_number_partitioning_qubo(numbers)

    for cd_type, noise in itertools.product(CD_TYPES, NOISE):
        print(f"{cd_type} {noise=}")
        results = defaultdict(list)
        results["n_layers"] = N_LAYERS
        for n_layers, n_try in itertools.product(N_LAYERS, N_RETRIES):
            succ = run_optimizer(qubo, n_layers, cd_type, noise)
            print(f"{n_layers:>3} layers: {succ:.2%}")

            name = f"try{n_try}"
            results[name].append(succ)

        filename = f"{len(qubo)+1}numbers_{cd_type}_noise{noise:.3f}.json"
        filepath = Path(__file__).parent / "data" / filename
        json_text = json.dumps(results)
        filepath.write_text(json_text, encoding="utf-8")

def run_optimizer(qubo: QUBO, n_layers: int, cd_type: str, noise: float) -> float:
    cd_type = str(cd_type).strip().lower()
    cd_type = None if cd_type == "no cd" else cd_type

    noise_model = build_noise_model(noise)
    perfect_device = qml.device("default.mixed", shots=10_000, wires=len(qubo))
    noisy_device = qml.add_noise(perfect_device, noise_model)


    optimizer = CDQOptimizer(n_layers, noisy_device, cd_type)

    if cd_type == "no cd":
        counts = optimizer.sample_qubo(qubo)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            counts = optimize_cdqo_gridsearch(optimizer, qubo)

    return calc_succ_prob(qubo, counts)  


def build_noise_model(noise: float) -> qml.NoiseModel:
    cond1 = qml.BooleanFn(lambda x: True)
    cond2 = qml.BooleanFn(lambda x: True)
    noise1 = qml.noise.partial_wires(qml.AmplitudeDamping, noise)
    noise2 = qml.noise.partial_wires(qml.DepolarizingChannel, noise)
    return qml.NoiseModel({cond1:noise1, cond2:noise2})

def calc_succ_prob(qubo: QUBO, counts: dict[str, int]) -> float:
    optimal_shots = 0
    for bitstring, count in counts.items():
        bits = np.fromiter(bitstring, np.uint8)
        if qubo.evaluate(bits) == 0:
            optimal_shots += count
    return optimal_shots / sum(counts.values())


if __name__ == "__main__":
    for i in range(3, len(NUMBERS) + 1):
        run(NUMBERS[:i])
