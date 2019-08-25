import pytest

import simulacra as si


class DummySim(si.Simulation):
    def run(self):
        """run method docstring"""
        pass


class DummySpec(si.Specification):
    simulation_type = DummySim


@pytest.fixture(scope="function")
def spec():
    return DummySpec("dummy")


@pytest.fixture(scope="function")
def sim(spec):
    return spec.to_sim()
