import pytest

import numpy as np

import simulacra as si


@pytest.fixture(scope="function")
def arr():
    return np.array([0, 2, 4, 6])


def test_find_nearest_with_target_in_array(arr):
    index, value, target = si.utils.find_nearest_entry(arr, 4)

    assert index == 2
    assert value == 4
    assert target == 4


def test_find_nearest_with_target_not_in_array(arr):
    index, value, target = si.utils.find_nearest_entry(arr, 3.9)

    assert index == 2
    assert value == 4
    assert target == 3.9
