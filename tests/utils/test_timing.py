import pytest

import simulacra as si


def test_block_timer():
    timer = si.utils.BlockTimer()

    assert timer.proc_time_start is None
    assert timer.proc_time_end is None
    assert timer.proc_time_elapsed is None
    assert timer.wall_time_start is None
    assert timer.wall_time_end is None
    assert timer.wall_time_elapsed is None

    with si.utils.BlockTimer() as timer:
        assert timer.proc_time_start is not None
        assert timer.proc_time_end is None
        assert timer.proc_time_elapsed is None
        assert timer.wall_time_start is not None
        assert timer.wall_time_end is None
        assert timer.wall_time_elapsed is None

    assert timer.proc_time_start is not None
    assert timer.proc_time_end is not None
    assert timer.proc_time_elapsed is not None
    assert timer.wall_time_start is not None
    assert timer.wall_time_end is not None
    assert timer.wall_time_elapsed is not None
