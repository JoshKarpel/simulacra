import pytest

import simulacra as si
import simulacra.cluster as clu


def test_two_fixed_parameters():
    a = clu.Parameter("a", [0, 1])
    b = clu.Parameter("b", [True, False])
    parameters = [a, b]

    d = clu.expand_parameters(parameters)

    assert d == [{"a": [0, 1], "b": [True, False]}]


def test_expandable_and_fixed_parameter():
    a = clu.Parameter("a", [0, 1])
    b = clu.Parameter("b", [True, False], expandable=True)
    parameters = [a, b]

    d = clu.expand_parameters(parameters)

    assert d == [{"a": [0, 1], "b": True}, {"a": [0, 1], "b": False}]


def test_two_expandable_parameters_with_sized_values():
    a = clu.Parameter("a", [0, 1], expandable=True)
    b = clu.Parameter("b", [True, False], expandable=True)
    parameters = [a, b]

    d = clu.expand_parameters(parameters)

    assert d == [
        {"a": 0, "b": True},
        {"a": 0, "b": False},
        {"a": 1, "b": True},
        {"a": 1, "b": False},
    ]


def test_expandable_parameters_with_iterator():
    a = clu.Parameter("a", range(2), expandable=True)
    b = clu.Parameter("b", [True, False], expandable=True)
    parameters = [a, b]

    d = clu.expand_parameters(parameters)

    assert d == [
        {"a": 0, "b": True},
        {"a": 0, "b": False},
        {"a": 1, "b": True},
        {"a": 1, "b": False},
    ]
