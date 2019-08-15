import pytest

import numpy as np

import simulacra as si


def test_dict_to_arrays():
    d = {1: 2, 3: 4, 5: -10}
    k, v = si.utils.dict_to_arrays(d)

    target_k, target_v = np.array([1, 3, 5]), np.array([2, 4, -10])

    assert np.all(k == target_k)
    assert np.all(v == target_v)


def test_dict_to_arrays_with_key():
    d = {1: 2, 3: 4, 5: -10}
    k, v = si.utils.dict_to_arrays(d, key=lambda x: -x[0])

    target_k, target_v = np.array([5, 3, 1]), np.array([-10, 4, 2])

    assert np.all(k == target_k)
    assert np.all(v == target_v)
