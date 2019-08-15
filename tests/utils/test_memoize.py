import pytest

import simulacra as si


@pytest.fixture(scope="function")
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


def test_cached_property(mocker):
    func = mocker.MagicMock(return_value="foo")

    class Foo:
        @si.utils.cached_property
        def prop(self):
            return func()

    f = Foo()
    assert f.prop == "foo"
    assert "prop" in f.__dict__

    f.prop
    f.prop
    f.prop
    f.prop

    assert func.call_count == 1  # only called once (i.e., was cached)

    del f.prop  # delete to reset caching

    assert "prop" not in f.__dict__
    assert f.prop == "foo"  # call again
    assert func.call_count == 2
    assert "prop" in f.__dict__
