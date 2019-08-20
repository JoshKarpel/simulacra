import pytest

import simulacra as si


def test_spec_absorbs_extra_kwargs():
    s = si.Specification("foobar", extra="joe", bing="baz")

    assert s._extra_attr_keys == {"extra", "bing"}
    assert s.extra == "joe"
    assert s.bing == "baz"
