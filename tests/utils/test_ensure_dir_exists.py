import os

import simulacra as si


def test_ensure_dir_exists_on_dirname(tmpdir):
    dirname = 'foo'
    dirpath = os.path.join(tmpdir, dirname)

    si.utils.ensure_dir_exists(dirpath)

    assert os.path.exists(dirpath)


def test_ensure_dir_exists_on_filename(tmpdir):
    dirname = 'foo'
    filename = 'bar.py'
    dirpath = os.path.join(tmpdir, dirname)
    create_path = os.path.join(dirpath, filename)

    si.utils.ensure_dir_exists(create_path)

    assert os.path.exists(dirpath)
    assert not os.path.exists(create_path)
