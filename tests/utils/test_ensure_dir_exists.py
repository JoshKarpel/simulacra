import os

import simulacra as si


def test_ensure_parents_exist(tmpdir):
    dirname = "foo"
    filename = "bar.py"
    dirpath = os.path.join(tmpdir, dirname)
    create_path = os.path.join(dirpath, filename)

    si.utils.ensure_parents_exist(create_path)

    assert os.path.exists(dirpath)
    assert not os.path.exists(create_path)
