from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dqao.benchmarking import (
    build_number_partitioning_qubo,
    decode_number_partitioning_bits,
    random_number_partitioning_qubo,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def test_build_number_partitioning_qubo() -> None:
    numbers = [1, 2, 3]
    qubo = build_number_partitioning_qubo(numbers)
    opt_bits, opt_value = qubo.brute_force()

    assert opt_value == 0
    np.testing.assert_array_equal(opt_bits, [1, 0])


@pytest.mark.parametrize("n_numbers", range(2, 6))
def test_random_number_partitioning_qubo(n_numbers: int) -> None:
    for _ in range(100):
        numbers, qubo = random_number_partitioning_qubo(n_numbers)
        opt_bits, opt_value = qubo.brute_force()

        partition1, partition2 = decode_number_partitioning_bits(numbers, opt_bits)

        assert len(numbers) == n_numbers
        assert opt_value == 0
        assert len(partition1) + len(partition2) == n_numbers
        assert np.sum(partition1) == np.sum(partition2)


def test_random_number_partitioning_problem_seed() -> None:
    numbers1, qubo1 = random_number_partitioning_qubo(10, seed=42)
    numbers2, qubo2 = random_number_partitioning_qubo(10, seed=42)
    numbers3, qubo3 = random_number_partitioning_qubo(10, seed=43)

    np.testing.assert_array_equal(numbers1, numbers2)
    assert any(numbers1 != numbers3)

    qubo1 == qubo2
    qubo1 != qubo3


@pytest.mark.parametrize(
    ("numbers", "bits", "expected_p1", "expected_p2"),
    [([1, 3, 2], [0, 1], [1, 2], [3]), ([1], [], [1], [])],
)
def test_decode_number_partitioning_bits(
    numbers: ArrayLike, bits: ArrayLike, expected_p1: ArrayLike, expected_p2: ArrayLike
) -> None:
    partition1, partition2 = decode_number_partitioning_bits(numbers, bits)

    np.testing.assert_array_equal(partition1, expected_p1)
    np.testing.assert_array_equal(partition2, expected_p2)
