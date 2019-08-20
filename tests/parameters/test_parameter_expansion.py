import pytest

import simulacra as si


def test_two_fixed_parameters():
    a = si.Parameter("a", [0, 1])
    b = si.Parameter("b", [True, False])
    parameters = [a, b]

    d = si.expand_parameters(parameters)

    assert d == [{"a": [0, 1], "b": [True, False]}]


def test_expandable_and_fixed_parameter():
    a = si.Parameter("a", [0, 1])
    b = si.Parameter("b", [True, False], expandable=True)
    parameters = [a, b]

    d = si.expand_parameters(parameters)

    assert d == [{"a": [0, 1], "b": True}, {"a": [0, 1], "b": False}]


def test_two_expandable_parameters_with_sized_values():
    a = si.Parameter("a", [0, 1], expandable=True)
    b = si.Parameter("b", [True, False], expandable=True)
    parameters = [a, b]

    d = si.expand_parameters(parameters)

    assert d == [
        {"a": 0, "b": True},
        {"a": 0, "b": False},
        {"a": 1, "b": True},
        {"a": 1, "b": False},
    ]


def test_expandable_parameters_with_iterator():
    a = si.Parameter("a", range(2), expandable=True)
    b = si.Parameter("b", [True, False], expandable=True)
    parameters = [a, b]

    d = si.expand_parameters(parameters)

    assert d == [
        {"a": 0, "b": True},
        {"a": 0, "b": False},
        {"a": 1, "b": True},
        {"a": 1, "b": False},
    ]
