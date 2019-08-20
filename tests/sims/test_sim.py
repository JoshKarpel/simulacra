import pytest

import time

import simulacra as si


def test_base_simulation_cannot_be_instantiated_because_abstract():
    with pytest.raises(TypeError):
        si.Specification("dummy").to_sim()


class DummySim(si.Simulation):
    def run(self):
        pass


class DummySpec(si.Specification):
    simulation_type = DummySim


def test_can_instantiate_dummy_sim():
    DummySpec("dummy").to_sim()


@pytest.fixture(scope="function")
def sim():
    return DummySpec("sim").to_sim()


def test_fresh_status_is_initialized(sim):
    assert sim.status is si.Status.INITIALIZED


def test_set_to_current_status(sim):
    old_status = sim.status
    sim.status = sim.status
    assert sim.status == old_status


@pytest.mark.parametrize(
    "not_status",
    [True, False, "running", "initialized", 0, 5, 2.5, [], (), {}, set(), 1j],
)
def test_cannot_set_non_status_status(sim, not_status):
    with pytest.raises(TypeError):
        sim.status = not_status


def test_start_time_is_set_after_first_run(sim):
    sim.status = si.Status.RUNNING

    assert sim.start_time is not None


def test_latest_run_time_is_set_after_first_run(sim):
    sim.status = si.Status.RUNNING

    assert sim.latest_run_time is not None


def test_start_time_and_latest_run_time_are_same_after_first_run(sim):
    sim.status = si.Status.RUNNING

    assert sim.start_time == sim.latest_run_time


def test_latest_run_time_updates_after_pause_then_run(sim):
    sim.status = si.Status.RUNNING
    first_run_time = sim.latest_run_time
    time.sleep(0.001)
    sim.status = si.Status.PAUSED
    sim.status = si.Status.RUNNING

    assert first_run_time < sim.latest_run_time


def test_end_time_set_after_finished(sim):
    sim.status = si.Status.FINISHED

    assert sim.end_time is not None


def test_elapsed_time_set_after_finished(sim):
    sim.status = si.Status.FINISHED

    assert sim.elapsed_time is not None
