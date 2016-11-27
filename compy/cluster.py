import collections
import datetime as dt
import json
import logging
import os
import sys
import posixpath
import stat
import subprocess
import time

import paramiko

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CmdOutput = collections.namedtuple('CmdOutput', ['stdin', 'stdout', 'stderr'])


class ClusterInterface:
    """
    A class for communicating with a user's home directory on a remote machine (typically a cluster). Should be used as a context manager.

    The remote home directory should look like:

    home/
    |-- backend/
    |-- jobs/
        |-- job1/
        |-- job2/
    """

    def __init__(self, remote_host, username, key_path,
                 local_mirror_root = 'cluster_mirror', remote_sep = '/'):
        self.remote_host = remote_host
        self.username = username
        self.key_path = key_path

        self.local_mirror_root = local_mirror_root
        self.remote_sep = remote_sep

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
        return 'Interface to {} as {}'.format(self.remote_host, self.username)

    def __repr__(self):
        return '{}(hostname = {}, username = {})'.format(self.__class__.__name__, self.remote_host, self.username)

    @utils.cached_property
    def local_home_dir(self):
        local_home = os.path.join(self.local_mirror_root, *self.remote_home_dir.split(self.remote_sep))

        return local_home

    def cmd(self, cmd_list):
        """Run a list of commands sequentially on the remote host. Each command list begins in a totally fresh environment."""
        cmd_list = ['. ~/.profile', '. ~/.bash_profile'] + cmd_list  # run the remote bash profile to pick up settings
        cmd = ';'.join(cmd_list)
        stdin, stdout, stderr = self.ssh.exec_command(cmd)

        return CmdOutput(stdin, stdout, stderr)

    @utils.cached_property
    def remote_home_dir(self):
        cmd_output = self.cmd(['pwd'])  # print name of home dir to stdout

        home_path = str(cmd_output.stdout.readline()).strip('\n')  # extract path of home dir from stdout

        logger.debug('Got home directory for {} on {}: {}'.format(self.username, self.remote_host, home_path))

        return home_path

    @property
    def job_status(self):
        cmd_output = self.cmd(['condor_q'])

        status = cmd_output.stdout.readlines()

        status_str = 'Job Status:\n' + ''.join(status[1:])

        return status_str

    def remote_path_to_local(self, remote_path):
        return os.path.join(self.local_mirror_root, *remote_path.split(self.remote_sep))

    def get_file(self, remote_path, local_path, remote_stat = None, preserve_timestamps = True):
        utils.ensure_dir_exists(local_path)

        self.ftp.get(remote_path, local_path)

        if preserve_timestamps:
            if remote_stat is None:
                remote_stat = self.ftp.lstat(remote_path)
            os.utime(local_path, (remote_stat.st_atime, remote_stat.st_mtime))

        logger.debug('{}   <--   {}'.format(local_path, remote_path))

    # def put_file(self, local_path, remote_path, preserve_timestamps = True):
    #     #  TODO: ensure remote dir
    #     self.ftp.put(local_path, remote_path)
    #
    #     if preserve_timestamps:
    #         pass
    #         #  TODO: this
    #
    #     logger.debug('{}   -->   {}'.format(local_path, remote_path))

    def is_file_synced(self, remote_stat, local_path):
        if os.path.exists(local_path):
            local_stat = os.stat(local_path)
            if local_stat.st_size == remote_stat.st_size and local_stat.st_mtime == remote_stat.st_mtime:
                return True

        return False

    def mirror_file(self, remote_path, remote_stat):
        local_path = self.remote_path_to_local_path(remote_path)
        if not self.is_file_synced(remote_stat, local_path):
            self.get_file(remote_path, local_path, remote_stat = remote_stat, preserve_timestamps = True)

    def walk_remote_path(self, remote_path, func_on_dirs = None, func_on_files = None, exclude_hidden = True, blacklist_dir_names = None, whitelist_file_ext = None):
        """
        Walk a remote directory starting at the given path.

        The functions func_on_dirs and func_on_files are passed the full path to the remote file and the ftp.stat of that file.
        """
        if func_on_dirs is None:
            func_on_dirs = lambda *args: None
        if func_on_files is None:
            func_on_files = lambda *args: None

        # make sure each whitelisted file extension actually looks like a file extension
        cleaned = []
        for ext in whitelist_file_ext:
            if ext[0] != '.':
                clean = '.' + ext
            else:
                clean = ext
            cleaned.append(clean)
        whitelist_file_ext = tuple(cleaned)

        path_count = 0

        def walk(remote_path):
            for remote_stat in self.ftp.listdir_attr(remote_path):
                full_remote_path = posixpath.join(remote_path, remote_stat.filename)

                logger.debug('Checking remote path {}'.format(full_remote_path))

                nonlocal path_count
                path_count += 1
                status_str = '\rPaths Found: {}'.format(path_count).ljust(25)
                status_str += '  |  '
                status_str += 'Current Path: {}'.format(full_remote_path).ljust(100)
                print(status_str, end = '')

                if not exclude_hidden or remote_stat.filename[0] != '.':
                    if stat.S_ISDIR(remote_stat.st_mode) and remote_stat.filename not in blacklist_dir_names:
                        func_on_dirs(full_remote_path, remote_stat)

                        logger.debug('Walking remote dir {}'.format(full_remote_path))
                        walk(full_remote_path)

                    elif stat.S_ISREG(remote_stat.st_mode) and full_remote_path.endswith(whitelist_file_ext):
                        func_on_files(full_remote_path, remote_stat)

        walk(remote_path)
        print()

    def mirror_remote_home_dir(self, blacklist_dir_names = ('python', 'build_python', 'ionization'), whitelist_file_ext = ('.txt', '.log', '.par', '.sim')):
        start_time = dt.datetime.now()
        logger.info('Mirroring remote home directory')

        self.walk_remote_path(self.remote_home_dir, func_on_files = self.mirror_file, blacklist_dir_names = blacklist_dir_names, whitelist_file_ext = whitelist_file_ext)

        end_time = dt.datetime.now()
        logger.info('Mirroring complete. Elapsed time: {}'.format(end_time - start_time))

        # @property
        # def local_job_names(self):
        #     return (job_dir for job_dir in os.listdir(os.path.join(self.local_home_dir, 'jobs')))
        #
        # def process_job(self, job_name, individual_processing = False):
        #     job_dir_path = os.path.join(self.local_home_dir, 'jobs', job_name)
        #     try:
        #         jp = JobProcessor.load(os.path.join(job_dir_path, '{}.job'.format(job_name)))
        #     except (FileNotFoundError, EOFError, ImportError):
        #         job_info_path = os.path.join(job_dir_path, 'info.json')
        #         with open(job_info_path, mode = 'r') as f:
        #             job_info = json.load(f)
        #         jp = job_info['job_processor'](job_dir_path)
        #
        #     jp.process_job(individual_processing = individual_processing)
        #
        # def process_jobs(self, individual_processing = False):
        #     start_time = dt.datetime.now()
        #     logger.info('Processing jobs')
        #
        #     for job_name in self.local_job_names:
        #         self.process_job(job_name = job_name, individual_processing = individual_processing)
        #
        #     end_time = dt.datetime.now()
        #     logger.info('Processing complete. Elapsed time: {}'.format(end_time - start_time))
        #
        # def sync_process_loop(self, wait_after_success = dt.timedelta(hours = 1), wait_after_failure = dt.timedelta(minutes = 1)):
        #     latest_sync_time = None
        #     while True:
        #         if latest_sync_time is None or dt.datetime.now() - latest_sync_time > wait_after_success:
        #             try:
        #                 start_time = dt.datetime.now()
        #                 logger.info('Beginning automatic synchronization and processing')
        #
        #                 logger.info(self.job_status)
        #
        #                 self.mirror_remote_home_dir()
        #                 self.process_jobs()
        #
        #                 end_time = dt.datetime.now()
        #                 logger.info('Synchronization and processing complete. Elapsed time: {}'.format(end_time - start_time))
        #
        #                 latest_sync_time = end_time
        #                 logger.info('Next automatic synchronization attempt after {}'.format(latest_sync_time + wait_after_success))
        #             except (FileNotFoundError, PermissionError, TimeoutError) as e:
        #                 logger.exception('Exception encountered')
        #                 logger.warning('Automatic synchronization attempt failed')
        #
        #         time.sleep(wait_after_failure.total_seconds())


def ask_for_input(question, default = None, cast_to = str):
    """Ask for input from the user, with a default value, and call cast_to on it before returning it."""
    input_str = input(question + ' [Default: {}]: '.format(default))

    trimmed = input_str.replace(' ', '')
    if trimmed == '':
        out = cast_to(default)
    else:
        out = cast_to(trimmed)

    logger.debug('Got input from cmd line: {} [type: {}]'.format(out, type(out)))

    # input_str = input(question + ' [Default: {}]: '.format(default))
    #
    # trimmed = (s.strip() for s in input_str.split(','))

    return out


def create_job_dirs(job_name):
    print('Creating job directory and subdirectories...')

    os.makedirs(job_name, exist_ok = True)
    os.chdir(job_name)
    os.makedirs('inputs/', exist_ok = True)
    os.makedirs('outputs/', exist_ok = True)
    os.makedirs('logs/', exist_ok = True)


def save_parameters(parameters):
    print('Saving Parameters...')

    for parameter in parameters:
        parameter.save(target_dir = 'inputs/')


def write_parameter_info_to_file(parameters):
    print('Writing Parameters info to file...')
    with open('parameters.txt', 'w') as file:
        for parameter in parameters:
            file.write(parameter.info())
            file.write('\n' + ('-' * 10) + '\n')  # line between Parameters


CHTC_SUBMIT_STRING = """universe = vanilla
log = logs/cluster_$(Cluster).log
#
executable = /home/karpel/backend/run_sim.sh
arguments = $(Process)
#
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = /home/karpel/backend/compy.tar.gz, /home/karpel/backend/run_sim.py, inputs/$(Process).par, http://proxy.chtc.wisc.edu/SQUID/karpel/python.tar.gz
transfer_output_remaps = "$(Process).sim = outputs/$(Process).sim ; $(Process).log = logs/$(Process).log ; $(Process).mp4 = outputs/$(Process).mp4"
#
+JobBatchName = "{}"
#
+is_resumable = {}
+WantGlidein = {}
+WantFlocking = {}
#
on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
#
request_cpus = 1
request_memory = {}GB
request_disk = {}GB
#
queue {}"""


def format_chtc_submit_string(job_name, parameter_count, memory = 4, disk = 4, checkpoints = True):
    if checkpoints:
        check = 'true'
    else:
        check = 'false'

    submit_string = CHTC_SUBMIT_STRING.format(job_name, check, check, check, memory, disk, parameter_count)

    return submit_string


def specification_check(parameters):
    print('Generated {} Parameters'.format(len(parameters)))

    print('-' * 20)
    print(parameters[0].info())
    print('-' * 20)

    parameter_check = input('Does the first Specification look correct? (y/[n]) ')
    parameter_check = parameter_check.strip(' ')
    if parameter_check != 'y':
        print('Aborting job creation...')
        sys.exit(0)


def submit_check(submit_string):
    print('-' * 20)
    print(submit_string)
    print('-' * 20)

    submit_check = input('Does the submit file look correct? (y/[n]) ')
    submit_check = submit_check.strip(' ')
    if submit_check != 'y':
        print('Aborting job creation...')
        sys.exit(0)


def write_submit_file(submit_string):
    print('Saving submit file...')

    with open('submit_job.sub', mode = 'w') as file:
        file.write(submit_string)


def submit_job():
    print('Submitting job...')

    subprocess.run(['condor_submit', 'submit_job.sub'])


class JobProcessor(utils.Beet):
    def __init__(self, job_name):
        super(JobProcessor, self).__init__(job_name)

    def save(self, target_dir = None, file_extension = '.job'):
        return super(JobProcessor, self).save(target_dir = target_dir, file_extension = file_extension)

    @classmethod
    def load(cls, file_path):
        return super(JobProcessor, cls).load(file_path)

    def load_sim(self):
        raise NotImplementedError

    def process_job(self, individual_processing = False):
        raise NotImplementedError

    def write_to_csv(self):
        raise NotImplementedError
