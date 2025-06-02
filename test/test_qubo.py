from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import digital_quantum_optimization as dqo

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@pytest.fixture(name="qubo")
def qubo_fixture() -> dqo.QUBO:
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    offset = 10
    return dqo.QUBO(matrix, offset)


@pytest.mark.parametrize(
    ("bits", "expected_value"),
    [
        ([0, 0, 0], 10),
        ([0, 0, 1], 19),
        ([0, 1, 0], 15),
        ([0, 1, 1], 38),
        ([1, 0, 0], 11),
        ([1, 0, 1], 30),
        ([1, 1, 0], 22),
        ([1, 1, 1], 55),
    ],
)
def test_evaluate(qubo: dqo.QUBO, bits: ArrayLike, expected_value: float) -> None:
    assert qubo.evaluate(bits) == expected_value


def test_to_lenz_ising(qubo: dqo.QUBO) -> None:
    expected_interactions = [[0, 1.5, 2.5], [0, 0, 3.5], [0, 0, 0]]
    expected_bias_terms = [-4.5, -7.5, -10.5]
    expected_offset = 25

    interactions, bias_terms, offset = qubo.to_lenz_ising()

    np.testing.assert_array_equal(interactions, expected_interactions)
    np.testing.assert_array_equal(bias_terms, expected_bias_terms)
    assert offset == expected_offset


def test_brute_force(qubo: dqo.QUBO) -> None:
    optimal_bits, optimal_value = qubo.brute_force()

    expected_value = 10
    assert optimal_value == expected_value
    np.testing.assert_equal(optimal_bits, [0, 0, 0])


def test_brute_force2() -> None:
    matrix = [[1, 2, -10], [0, -1, 3], [0, 0, 4]]
    offset = -10
    qubo = dqo.QUBO(matrix, offset)
    optimal_bits, optimal_value = qubo.brute_force()

    expected_value = -15

    assert optimal_value == expected_value
    np.testing.assert_equal(optimal_bits, [1, 0, 1])


def test_len(qubo: dqo.QUBO) -> None:
    qubo_size = 3
    assert len(qubo) == qubo_size


@pytest.mark.parametrize(
    ("qubo1", "qubo2"),
    [
        (dqo.QUBO([], 0), dqo.QUBO(np.empty(0), 0.0)),
        (dqo.QUBO([[1, 2], [3, 4]], 5), dqo.QUBO([[1, 2], [3, 4]], 5)),
        (dqo.QUBO([[1, 2], [-2, -1]]), dqo.QUBO([[1, 0], [0, -1]])),
    ],
)
def test_eq(qubo1: dqo.QUBO, qubo2: dqo.QUBO) -> None:
    assert qubo1 == qubo2


@pytest.mark.parametrize(
    ("qubo1", "qubo2"),
    [
        (dqo.QUBO([], 0), dqo.QUBO([], 1)),
        (dqo.QUBO([[1, 2], [3, 4]], 5), dqo.QUBO([[1, 2], [3, 4]], 6)),
        (dqo.QUBO([[1, 2], [3, 4]]), dqo.QUBO([[1, 2], [3, -4]])),
        (dqo.QUBO(np.zeros((2, 2))), dqo.QUBO(np.zeros((3, 3)))),
    ],
)
def test_nq(qubo1: dqo.QUBO, qubo2: dqo.QUBO) -> None:
    assert qubo1 != qubo2
