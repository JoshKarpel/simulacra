import pytest

import simulacra as si


def test_find_or_init(spec, tmp_path):
    sim = spec.to_sim()
    sim.save(target_dir=tmp_path)

    again = si.find_sim_or_init(spec, search_dir=tmp_path)

    assert sim is not again
    assert sim == again


def test_run_from_cache_gets_same_uid(spec, tmp_path):
    sim = si.run_from_cache(spec, cache_dir=tmp_path)
    again = si.run_from_cache(spec, cache_dir=tmp_path)

    assert sim is not again
    assert sim == again


def test_run_from_cache_latest_run_time_same(spec, tmp_path):
    sim = si.run_from_cache(spec, cache_dir=tmp_path)
    again = si.run_from_cache(spec, cache_dir=tmp_path)

    assert sim.latest_run_time == again.latest_run_time
