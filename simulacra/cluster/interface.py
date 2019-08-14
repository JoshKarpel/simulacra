import logging
import os
import hashlib
import posixpath
import stat
import collections
import functools
from typing import Iterable, Optional, Callable

import paramiko

from .. import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CmdOutput = collections.namedtuple('CmdOutput', ['stdin', 'stdout', 'stderr'])


class ClusterInterface:
    """
    A class for communicating with a cluster's submit node via SSH and FTP.
    Should be used as a context manager.
    """

    def __init__(
        self,
        hostname: str,
        username: str,
        key_path: str,
        local_mirror_root: str = 'mirror',
    ):
        """
        Parameters
        ----------
        hostname
            The hostname of the remote host.
        username
            The username to log in with.
        key_path
            The path to the SSH key file that corresponds to the `username`.
        local_mirror_root
            The name to give the root directory of the local mirror.
        """
        self.remote_host = hostname
        self.username = username
        self.key_path = key_path

        self.local_mirror_root = local_mirror_root

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ftp = None

    def __enter__(self):
        """Open the SSH and FTP connections."""
        self.connect()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the SSH and FTP connections."""
        self.close()

    def connect(self):
        """Open the connection to the remote host."""
        self.ssh.connect(self.remote_host, username = self.username, key_filename = self.key_path)
        self.ftp = self.ssh.open_sftp()

        logger.info(f'Opened connection to {self.username}@{self.remote_host}')

    def close(self):
        """Close the connection to the remote host."""
        self.ftp.close()
        self.ssh.close()

        logger.info(f'Closed connection to {self.username}@{self.remote_host}')

    def __str__(self):
        return f'Interface to {self.remote_host} as {self.username}'

    def __repr__(self):
        return f'{self.__class__.__name__}(hostname = {self.remote_host}, username = {self.username})'

    @property
    def local_home_dir(self):
        local_home = os.path.join(self.local_mirror_root, *self.remote_home_dir.split('/'))

        return local_home

    def cmd(self, *cmds: Iterable[str]):
        """Run a list of commands sequentially on the remote host. Each command list begins in a totally fresh environment."""
        cmd_list = ['. ~/.profile', '. ~/.bash_profile'] + list(cmds)  # run the remote bash profile to pick up settings
        cmd = ';'.join(cmd_list)
        stdin, stdout, stderr = self.ssh.exec_command(cmd)

        return CmdOutput(stdin, stdout, stderr)

    @utils.cached_property
    def remote_home_dir(self):
        cmd_output = self.cmd('pwd')  # print name of home dir to stdout

        home_path = str(cmd_output.stdout.readline()).strip('\n')  # extract path of home dir from stdout

        logger.debug('Got home directory for {}@{}: {}'.format(self.username, self.remote_host, home_path))

        return home_path

    def remote_path_to_local_path(self, remote_path, local_root):
        """Return the local path corresponding to a remote path."""
        return os.path.join(local_root, *remote_path.split('/'))

    def get_file(
        self,
        remote_path: str,
        local_path: str,
        remote_stat = None,
        preserve_timestamps: bool = True,
    ):
        """
        Download a file from the remote machine to the local machine.

        Parameters
        ----------
        remote_path : :class:`str`
            The remote path to download.
        local_path : :class:`str`
            The local path to place the downloaded file.
        remote_stat
            The stat of the remote path (optimization, this method will fetch it if not passed in).
        preserve_timestamps : :class:`bool`
            If ``True``, copy the modification timestamps from the remote file to the local file.
        """
        utils.ensure_parents_exist(local_path)

        self.ftp.get(remote_path, local_path)

        if preserve_timestamps:
            if remote_stat is None:
                remote_stat = self.ftp.lstat(remote_path)
            os.utime(local_path, (remote_stat.st_atime, remote_stat.st_mtime))

        logger.debug(f'{local_path}   <--   {remote_path}')

    def is_file_synced(
        self,
        remote_stat,
        local_path: str,
    ):
        """
        Determine whether a local file is the same as a remote file by checking the file size and modification times.

        Parameters
        ----------
        remote_stat
            The stat of the remote file.
        local_path : :class:`str`
            The path to the local file.

        Returns
        -------
        :class:`bool`
            ``True`` if the file is synced, ``False`` otherwise.
        """
        if os.path.exists(local_path):
            local_stat = os.stat(local_path)
            if local_stat.st_size == remote_stat.st_size and local_stat.st_mtime == remote_stat.st_mtime:
                return True

        return False

    def mirror_file(
        self,
        remote_path: str,
        remote_stat: str,
        local_root: str,
        force_download: bool = False,
        integrity_check: bool = True,
    ):
        """
        Mirror a remote file, only downloading it if it does not match a local copy at a derived local path name.

        File integrity is checked by comparing the MD5 hash of the remote and local files.

        Parameters
        ----------
        remote_path
            The remote path to mirror.
        remote_stat
            The stat of the remote path.
        local_root
            The local directory to use as the root directory.
        force_download
            If ``True``, download the file even if it synced.
        integrity_check
            If ``True``, check that the MD5 hash of the remote and local files are the same, and redownload if they are not.

        Returns
        -------
        local_path : str
            The path to the local file.
        """
        local_path = self.remote_path_to_local_path(remote_path, local_root)
        if force_download or not self.is_file_synced(remote_stat, local_path):
            self.get_file(remote_path, local_path, remote_stat = remote_stat, preserve_timestamps = True)
            if integrity_check:
                output = self.cmd(f'md5sum {remote_path}')
                md5_remote = output.stdout.readline().split(' ')[0].strip()
                with open(local_path, mode = 'rb') as f:
                    md5_local = hashlib.md5()
                    md5_local.update(f.read())
                    md5_local = md5_local.hexdigest().strip()
                if md5_local != md5_remote:
                    logger.debug(f'MD5 hash on {self.remote_host} for file {remote_path} did not match local file at {local_path}, retrying')
                    self.mirror_file(remote_path, remote_stat, local_root = local_root, force_download = True)

        return local_path

    def walk_remote_path(
        self,
        remote_path,
        func_on_dirs: Optional[Callable] = None,
        func_on_files: Optional[Callable] = None,
        exclude_hidden: bool = True,
        blacklist_dir_names: Iterable[str] = (),
        whitelist_file_ext: Iterable[str] = (),
    ):
        """
        Walk a remote directory starting at the given path.

        The functions func_on_dirs and func_on_files are passed the full path to the remote file and the ftp.stat of that file.

        Parameters
        ----------
        remote_path
            The remote path to start walking from.
        func_on_dirs
            The function to call on directories (takes the directory file path as an argument).
        func_on_files
            The function to call on files (takes the file path as an argument).
        exclude_hidden
            Do not walk over hidden files or directories.
        blacklist_dir_names
            Do not walk over directories with these names.
        whitelist_file_ext
            Only walk over files with these extensions.

        Returns
        -------
        path_count : int
            The number of paths checked.
        """
        if func_on_dirs is None:
            func_on_dirs = lambda *args: None
        if func_on_files is None:
            func_on_files = lambda *args: None

        blacklist_dir_names = tuple(blacklist_dir_names)
        whitelist_file_ext = tuple(ext if ext.startswith('.') else f'.{ext}' for ext in whitelist_file_ext)

        path_count = 0
        longest_full_remote_path_len = 0

        def walk(remote_path: str):
            try:
                remote_stats = self.ftp.listdir_attr(remote_path)
                for remote_stat in remote_stats:
                    full_remote_path = posixpath.join(remote_path, remote_stat.filename)

                    logger.debug(f'Checking remote path {full_remote_path}')

                    # print a string that keeps track of the walked paths
                    nonlocal path_count, longest_full_remote_path_len
                    path_count += 1
                    longest_full_remote_path_len = max(longest_full_remote_path_len, len(full_remote_path))

                    status_str = f'\rPaths Found: {path_count}'.ljust(25) + f'|  Current Path: {full_remote_path}'.ljust(longest_full_remote_path_len + 20)
                    print(status_str, end = '')

                    if not exclude_hidden or remote_stat.filename[0] != '.':
                        if stat.S_ISDIR(remote_stat.st_mode) and remote_stat.filename not in blacklist_dir_names:
                            func_on_dirs(full_remote_path, remote_stat)

                            logger.debug(f'Walking remote dir {full_remote_path}')
                            walk(full_remote_path)

                        elif stat.S_ISREG(remote_stat.st_mode) and full_remote_path.endswith(whitelist_file_ext):
                            func_on_files(full_remote_path, remote_stat)
            except UnicodeDecodeError as e:
                logger.exception(f'Encountered unicode decode error while getting directory attributes for {remote_path}. This can happen if there is a unicode filename in the directory.')

        walk(remote_path)
        print()

        return path_count

    def mirror_dir(
        self,
        remote_dir: str = None,
        local_root: str = 'mirror',
        blacklist_dir_names: Iterable[str] = ('python', 'build_python', 'backend', 'logs'),
        whitelist_file_ext: Iterable[str] = ('.txt', '.json', '.spec', '.sim', '.pkl'),
    ):
        """
        Mirror a directory recursively.

        If no directory is given, mirror the remote home directory.

        Parameters
        ----------
        remote_dir
            The path to the remote directory to mirror.
        blacklist_dir_names
            Directories with these names will not be walked.
        local_root
            The directory to use as the root directory locally.
        whitelist_file_ext
            Only files with these file extensions will be transferred.
        """
        remote_dir = remote_dir or self.remote_home_dir

        logger.info(f'Mirroring remote dir {remote_dir}')

        with utils.BlockTimer() as timer:
            self.walk_remote_path(
                self.remote_home_dir,
                func_on_files = functools.partial(self.mirror_file, local_root = local_root),
                func_on_dirs = lambda d, _: utils.ensure_parents_exist(d),
                blacklist_dir_names = tuple(blacklist_dir_names),
                whitelist_file_ext = tuple(whitelist_file_ext),
            )

        logger.info(f'Mirroring complete. {timer}')
