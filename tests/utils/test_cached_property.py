import pytest

import simulacra as si


def test_cached_property(mocker):
    func = mocker.MagicMock(return_value = 'foo')

    class Foo:
        @si.utils.cached_property
        def prop(self):
            return func()

    f = Foo()
    assert f.prop == 'foo'
    assert 'prop' in f.__dict__

    f.prop
    f.prop
    f.prop
    f.prop

    assert func.call_count == 1  # only called once (i.e., was cached)

    del f.prop  # delete to reset caching

    assert 'prop' not in f.__dict__
    assert f.prop == 'foo'  # call again
    assert func.call_count == 2
    assert 'prop' in f.__dict__
