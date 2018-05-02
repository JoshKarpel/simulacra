import simulacra as si


def test_clone_changed():
    b = si.Beet('beet')

    b.foo = 0
    c = b.clone(foo = 1)

    assert c.foo != b.foo


def test_clone_unchanged():
    b = si.Beet('beet')

    b.foo = 0
    c = b.clone()

    assert c.foo == b.foo


def test_clone_changes_uuid():
    b = si.Beet('beet')

    c = b.clone()

    assert c.uuid != b.uuid


def test_cloned_beet_not_equal():
    b = si.Beet('beet')

    c = b.clone()

    assert c != b
