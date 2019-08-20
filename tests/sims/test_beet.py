import pytest

import simulacra as si


@pytest.fixture(scope="function")
def blank_beet():
    return si.Beet("beet")


def test_clone_changed(blank_beet):
    blank_beet.foo = 0
    c = blank_beet.clone(foo=1)

    assert c.foo != blank_beet.foo


def test_clone_unchanged(blank_beet):
    blank_beet.foo = 0
    c = blank_beet.clone()

    assert c.foo == blank_beet.foo


def test_clone_changes_uuid(blank_beet):
    c = blank_beet.clone()

    assert c.uuid != blank_beet.uuid


def test_cloned_beet_not_equal(blank_beet):
    c = blank_beet.clone()

    assert c != blank_beet


def test_is_hashable(blank_beet):
    assert hash(blank_beet)


def test_quality(blank_beet):
    assert blank_beet == blank_beet
    assert blank_beet != si.Beet("beet")


def test_can_be_put_in_set():
    assert {blank_beet}


def test_can_be_used_as_dict_key():
    assert {blank_beet: 0}


def test_round_trip_through_save(blank_beet, tmp_path):
    p = blank_beet.save(tmp_path)

    loaded = si.Beet.load(p)

    assert blank_beet == loaded
    assert blank_beet is not loaded
