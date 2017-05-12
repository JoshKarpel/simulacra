import os
import shutil


def tar(d):
    """Make a .tar.gz archive from the directory at path d, using the same name as d for the archive."""
    shutil.make_archive(d, format = 'gztar', base_dir = d)


def tar_dirs(dirs = None):
    """
    Make a .tar.gz archive for each dir path in dirs. If dirs = 'all', makes an archive from every dir in the cwd.
    """
    if dirs == 'all':
        dirs = (d for d in os.listdir(os.getcwd()) if os.path.isdir(d))

    for d in dirs:
        tar(d)


def untar(tar):
    raise NotImplementedError


def untar_dirs(tar):
    raise NotImplementedError


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description = 'Utility for tar operations using python.')
    parser.add_argument('-a', '-all', action = 'store_true', default = False, help = 'tar all dirs in the cwd')
