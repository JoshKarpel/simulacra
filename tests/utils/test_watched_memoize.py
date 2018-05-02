import pytest

import simulacra as si


class Foo:
    def __init__(self):
        self.w = True

    @si.utils.watched_memoize(lambda s: s.w)
    def memoized(self, x):
        return self.inner(x)

    def inner(self, x):
        raise NotImplementedError


@pytest.fixture(scope = 'function')
def watched_mock(mocker):
    foo = Foo()

    foo.inner = mocker.MagicMock()

    return foo


def test_watched_mock_is_only_called_once_for_repeated_args(watched_mock):
    watched_mock.memoized(1)
    watched_mock.memoized(1)
    watched_mock.memoized(1)
    watched_mock.memoized(1)
    watched_mock.memoized(1)

    assert watched_mock.inner.call_count == 1


def test_watched_mock_is_called_multiple_times_for_different_args(watched_mock):
    watched_mock.memoized(1)
    watched_mock.memoized(1)

    watched_mock.memoized(2)
    watched_mock.memoized(2)

    watched_mock.memoized(3)
    watched_mock.memoized(3)

    watched_mock.memoized(4)
    watched_mock.memoized(4)

    watched_mock.memoized(5)
    watched_mock.memoized(5)

    assert watched_mock.inner.call_count == 5


def test_watched_mock_resets_if_watched_value_changes(watched_mock):
    watched_mock.memoized(1)
    watched_mock.w = not watched_mock.w
    watched_mock.memoized(1)

    watched_mock.memoized(2)
    watched_mock.w = not watched_mock.w
    watched_mock.memoized(2)

    watched_mock.memoized(3)
    watched_mock.w = not watched_mock.w
    watched_mock.memoized(3)

    watched_mock.memoized(4)
    watched_mock.memoized(4)

    watched_mock.memoized(5)
    watched_mock.memoized(5)

    assert watched_mock.inner.call_count == 6 + 2  # 6 from first three pairs, two from last two pairs


def test_watched_memoize(mocker):
    func = mocker.MagicMock(return_value = 'foo')

    class Foo:
        def __init__(self):
            self.a = True
            self.counter = 0

        @si.utils.watched_memoize(lambda self: self.a)
        def method(self, x):
            func()
            self.counter += 1
            return self.counter

    f = Foo()
    assert f.method(0) == 1
    assert f.method(10) == 2
    assert f.method(10) == 2
    assert f.method(10) == 2
    assert f.method(10) == 2
    assert func.call_count == 2

    assert f.method(0) == 1
    assert f.method(3) == 3
    assert func.call_count == 3

    f.a = not f.a  # resets memo

    assert f.method(0) == 4
    assert func.call_count == 4
