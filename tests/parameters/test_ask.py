import pytest

import numpy as np

import simulacra as si


@pytest.mark.parametrize("answer", ["foobar", "wizbang", "floof", "1203", "15.194"])
def test_ask_for_input_str(mocker, answer):
    mocker.patch("simulacra.parameters.input", return_value=answer)

    assert si.ask_for_input("?") == answer


@pytest.mark.parametrize("answer", ["5", "1233", "102319"])
def test_ask_for_input_int(mocker, answer):
    mocker.patch("simulacra.parameters.input", return_value=answer)

    assert si.ask_for_input("?", cast_to=int) == int(answer)


@pytest.mark.parametrize("answer", ["5", "1233", "102319", "5091.59324", "10.24102"])
def test_ask_for_input_float(mocker, answer):
    mocker.patch("simulacra.parameters.input", return_value=answer)

    assert si.ask_for_input("?", cast_to=float) == float(answer)


@pytest.mark.parametrize(
    "answer", ["true", "TrUe", "t", "T", "yes", "y", "YeS", "1", "on"]
)
def test_ask_for_bool_with_true(mocker, answer):
    mocker.patch("simulacra.parameters.input", return_value=answer)

    assert si.ask_for_bool("?")


@pytest.mark.parametrize(
    "answer", ["false", "faLSE", "f", "F", "no", "NO", "n", "0", "off"]
)
def test_ask_for_bool_with_false(mocker, answer):
    mocker.patch("simulacra.parameters.input", return_value=answer)

    assert not si.ask_for_bool("?")


def test_ask_for_eval_with_np_linspace(mocker):
    mocker.patch("simulacra.parameters.input", return_value="np.linspace(0, 1, 100)")

    assert np.all(si.ask_for_eval("?") == np.linspace(0, 1, 100))


def test_ask_for_eval_with_units(mocker):
    mocker.patch("simulacra.parameters.input", return_value="10 * u.THz")

    assert si.ask_for_eval("?") == 10 * si.units.THz


def test_ask_for_choices_with_tuple(mocker):
    mocker.patch("simulacra.parameters.input", return_value="b")

    assert si.ask_for_choices("?", choices=("a", "b", "c")) == "b"


def test_ask_for_choices_with_dict(mocker):
    mocker.patch("simulacra.parameters.input", return_value="b")

    assert si.ask_for_choices("?", choices={"a": 0, "b": 1, "c": 2}) == 1


def test_ask_for_choices_with_tuple_default(mocker):
    mocker.patch("simulacra.parameters.input", return_value="")

    assert si.ask_for_choices("?", choices=("a", "b", "c")) == "a"


def test_ask_for_choices_with_dict_default(mocker):
    mocker.patch("simulacra.parameters.input", return_value="")

    assert si.ask_for_choices("?", choices={"a": 0, "b": 1, "c": 2}) == 0
