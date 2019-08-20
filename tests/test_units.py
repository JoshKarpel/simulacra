import simulacra.units as u


def test_get_unit_value():
    assert u.get_unit_value("km") == u.km


def test_get_unit_values():
    assert u.get_unit_values("m", "km", "mA") == (u.m, u.km, u.mA)


def test_get_unit_value_and_latex():
    assert u.get_unit_value_and_latex("m") == (u.m, r"\mathrm{m}")
