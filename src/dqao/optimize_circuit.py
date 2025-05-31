"""Helper methods to optimize parameters of the CounterDiabaticQuantumOptimizer."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, SupportsFloat

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dqao import QUBO
    from dqao import CounterDiabaticQuantumOptimizer as CDQOptimizer


def optimize_cdqo(cdqo: CDQOptimizer, qubo: QUBO) -> dict[str, int]:
    """Optimize the `end_time` hyperparameter of the CounterDiabaticQuantumOptimizer.

    Optimization is performed using Baysian optimization.

    Args:
        cdqo: CounterDiabaticQuantumOptimizer to optimze the `end_time` for.
        qubo: QUBO problem to optimize for.

    Returns:
        Result of the best parameter.

    """
    score_func = functools.partial(_score_function, cdqo, qubo)
    search_space = [Real(0.1, 10)]

    result = gp_minimize(
        score_func,
        search_space,
        x0=[1],
        n_initial_points=10,
        n_calls=100,
    )

    cdqo.end_time = result.x[0]
    return cdqo.sample_qubo(qubo)


def optimize_cdqo_gridsearch(cdqo: CDQOptimizer, qubo: QUBO) -> dict[str, int]:
    """Optimize the `end_time` hyperparameter of the CounterDiabaticQuantumOptimizer.

    Optimization is performed using a gridsearch.

    Args:
        cdqo: CounterDiabaticQuantumOptimizer to optimze the `end_time` for.
        qubo: QUBO problem to optimize for.

    Returns:
        Result of the best parameter.

    """
    best_score = float("inf")
    end_times = np.power(np.linspace(-1, 1, 20), 10)

    for end_time in end_times:
        score = _score_function(cdqo, qubo, [end_time])
        if score < best_score:
            best_score = score
            best_end_time = end_time

    cdqo.end_time = best_end_time
    return cdqo.sample_qubo(qubo)


def _score_function(
    cdqo: CDQOptimizer,
    qubo: QUBO,
    end_time: Sequence[SupportsFloat],
) -> float:
    cdqo.end_time = end_time[0]
    counts = cdqo.sample_qubo(qubo)

    score = 0
    for bitstring, count in counts.items():
        bits = [int(val) for val in bitstring]
        value = qubo.evaluate(bits)
        score += count * value

    return score
