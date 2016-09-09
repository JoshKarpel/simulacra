import os
import logging

import paramiko

from compy import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ClusterInterface:
    def __init__(self, remote_host, username, key_path):
        self.remote_host = remote_host
        self.username = username
        self.key_path = key_path

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ftp = None

        self.remote_home_dir = None

    def __enter__(self):
        """Open the SSH and FTP connections."""
        self.ssh.connect(self.remote_host, username = self.username, key_filename = self.key_path)
        self.ftp = self.ssh.open_sftp()

        logger.info('Opened connection to {} as {}'.format(self.remote_host, self.username))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the SSH and FTP connections."""
        self.ftp.close()
        self.ssh.close()

        logger.info('Closed connection to {} as {}'.format(self.remote_host, self.username))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '{}(hostname = {})'.format(self.__class__.__name__, self.remote_host)

    @utils.cached_property
    def local_home_dir(self):
        raise NotImplementedError

    def cmd(self, cmd_list):
        """Run a list of commands sequentially on the remote host. Each command list begins in a totally fresh environment."""
        cmd_list = ['. ~/.profile', '. ~/.bash_profile'] + cmd_list  # run the remote bash profile to pick up settings
        cmd = ';'.join(cmd_list)
        stdin, stdout, stderr = self.ssh.exec_command(cmd)

        return stdin, stdout, stderr

    @utils.cached_property
    def remote_home_dir(self):
        stdin, stdout, stderr = self.cmd(['pwd'])  # print name of home dir to stdout

        home_path = str(stdout.readline()).strip('\n')  # extract path of home dir from stdout

        logger.debug('Got home directory for {} on {}: {}'.format(self.username, self.remote_host, home_path))

        return home_path
