"""This modole contains generators for the number partitioning problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsInt

import numpy as np

import digital_quantum_optimization as dqo

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def build_number_partitioning_qubo(numbers: ArrayLike) -> dqo.QUBO:
    r"""Build a number partitioning problem for the provided `numbers`.

    The goal of the number paritioning problem is to find partition the numbers into
    two disjoint subsets such that the sum of the elements of the subsets are equal.

    In this representation, the first
    number is assigned to the first partition.

    The QUBO formulation is given by:

    .. math::

        \left(n_1/2 + \sum_{i=2}^N(x_i-1/2)n_i \right)^2.

    In the formuale above $n_i$ is number $i$ of the provided `numbers`, for
    $i \in \{1,2,\ldots,N\}$ and $x_i\in\{0,1\}$ are the binary desicion variables.

    The QUBO formulation was inspired by https://arxiv.org/abs/1302.5843 was used.

    Args:
        numbers: 1-D ArrayLike of numbers to partition.

    Returns:
        A number partitioning problem instance represented as a QUBO problem.

    """
    numbers = np.asarray(numbers)
    if numbers.ndim != 1:
        error_msg = (
            "'numbers' must be a 1-D ArrayLike, but after conversion to an ndarray the "
            f"array has dimension {numbers.ndim}"
        )
        raise ValueError(error_msg)
    total = np.sum(numbers[1:])
    matrix = np.outer(numbers[1:], numbers[1:]) - np.diag(
        (total - numbers[0]) * numbers[1:],
    )
    offset = (total - numbers[0]) ** 2 / 4
    return dqo.QUBO(matrix, offset)


def random_number_partitioning_qubo(
    n_numbers: SupportsInt,
    seed: SupportsInt | None = None,
) -> tuple[NDArray[np.int_], dqo.QUBO]:
    """Build a random number partitoning QUBO problem with `n_numbers` numbers.

    Builds an array if integer values such that there exists a partitioning which has
    equal sums. Using this array, a QUBO problem is build.

    Args:
        n_numbers: size of the numbers array.
        seed: seed to use when creating the random numbers list.

    Returns:
        Tuple containing an array of random integers and corresponding QUBO problem.
        The array of integers has at least one equal sum partitioning.

    """
    n_numbers = int(n_numbers)
    rng = np.random.default_rng(seed)

    size1 = rng.integers(1, n_numbers)
    size2 = n_numbers - size1

    numbers1 = rng.integers(-100, 101, size1)

    numbers2 = rng.integers(-100, 101, size2 - 1)
    last_number = np.sum(numbers1) - np.sum(numbers2)

    numbers = np.array([*numbers1, *numbers2, last_number])
    rng.shuffle(numbers)
    qubo = build_number_partitioning_qubo(numbers)

    return numbers, qubo


def decode_number_partitioning_bits(
    numbers: ArrayLike,
    bits: ArrayLike,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Decode a solution bitstring of the number partitioning QUBO formulation.

    Args:
        numbers: 1-D ArrayLike of numbers used to build the QUBO problem.
        bits: 1-D ArrayLike of bits to decode.

    Returns:
        Tuple containg the two partitions.

    """
    numbers = np.asarray(numbers)
    bits = np.asarray(bits, dtype=np.uint8)

    mask1 = [True, *(bits == 1)]
    mask2 = [False, *(bits == 0)]

    partition1 = numbers[mask1]
    partition2 = numbers[mask2]

    return partition1, partition2
