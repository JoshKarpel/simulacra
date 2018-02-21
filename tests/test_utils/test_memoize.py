import pytest

import simulacra as si


@pytest.fixture(scope = 'function')
def memoized_mock(mocker):
    func = mocker.MagicMock()

    memoized_func = si.utils.memoize(func)

    return func, memoized_func


def test_memoized_func_is_only_called_once_for_repeated_args(memoized_mock):
    func, memoized_func = memoized_mock

    memoized_func(1)
    memoized_func(1)
    memoized_func(1)
    memoized_func(1)
    memoized_func(1)

    assert func.call_count == 1


def test_memoized_func_is_called_multiple_times_for_different_args(memoized_mock):
    func, memoized_func = memoized_mock

    memoized_func(1)
    memoized_func(1)

    memoized_func(2)
    memoized_func(2)

    memoized_func(3)
    memoized_func(3)

    memoized_func(4)
    memoized_func(4)

    memoized_func(5)
    memoized_func(5)

    assert func.call_count == 5
