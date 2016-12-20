import collections
import datetime as dt
import json
import logging
import os
import sys
import posixpath
import stat
import subprocess
import itertools as it
import hashlib
from copy import copy, deepcopy
from collections import OrderedDict, defaultdict

from tqdm import tqdm
import numpy as np
import paramiko

from . import core
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

    @property
    def local_home_dir(self):
        local_home = os.path.join(self.local_mirror_root, *self.remote_home_dir.split(self.remote_sep))

        return local_home

    def cmd(self, *cmds):
        """Run a list of commands sequentially on the remote host. Each command list begins in a totally fresh environment."""
        cmd_list = ['. ~/.profile', '. ~/.bash_profile'] + list(cmds)  # run the remote bash profile to pick up settings
        cmd = ';'.join(cmd_list)
        stdin, stdout, stderr = self.ssh.exec_command(cmd)

        return CmdOutput(stdin, stdout, stderr)

    @utils.cached_property
    def get_remote_home_dir(self):
        cmd_output = self.cmd('pwd')  # print name of home dir to stdout

        home_path = str(cmd_output.stdout.readline()).strip('\n')  # extract path of home dir from stdout

        logger.debug('Got home directory for {} on {}: {}'.format(self.username, self.remote_host, home_path))

        return home_path

    def get_job_status(self):
        cmd_output = self.cmd('condor_q')

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

    def mirror_file(self, remote_path, remote_stat, force_download = False, integrity_check = True):
        local_path = self.remote_path_to_local(remote_path)
        if force_download or not self.is_file_synced(remote_stat, local_path):
            self.get_file(remote_path, local_path, remote_stat = remote_stat, preserve_timestamps = True)
            if integrity_check:
                output = self.cmd('openssl md5 {}'.format(remote_path))
                md5_remote = output.stdout.readline().split(' ')[1].strip()
                with open(local_path, mode = 'rb') as f:
                    md5_local = hashlib.md5()
                    md5_local.update(f.read())
                    md5_local = md5_local.hexdigest().strip()
                if md5_local != md5_remote:
                    logger.warning('MD5 hash on {} for file {} did not match local file at {}, retrying'.format(self.remote_host, remote_path, local_path))
                    self.mirror_file(remote_path, remote_stat, force_download = True)  # TODO: decide between force_download and simply deleting the local copy (it's corrupted anyway...)

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

    def mirror_remote_home_dir(self, blacklist_dir_names = ('python', 'build_python', 'ionization'), whitelist_file_ext = ('.txt', '.log', '.json', '.spec', '.sim')):
        start_time = dt.datetime.now()
        logger.info('Mirroring remote home directory')

        self.walk_remote_path(self.get_remote_home_dir(), func_on_files = self.mirror_file, blacklist_dir_names = blacklist_dir_names, whitelist_file_ext = whitelist_file_ext)

        end_time = dt.datetime.now()
        logger.info('Mirroring complete. Elapsed time: {}'.format(end_time - start_time))


class JobProcessorManager:
    def __init__(self, jobs_dir):
        self.jobs_dir = os.path.abspath(jobs_dir)

    @property
    def job_dir_names(self):
        return (name for name in sorted(os.listdir(self.jobs_dir)) if os.path.isdir(os.path.join(self.jobs_dir, name)))

    @property
    def job_dir_paths(self):
        return (os.path.abspath(os.path.join(self.jobs_dir, name)) for name in self.job_dir_names)

    def process_job(self, job_name, job_dir_path, individual_processing = False, force_reprocess = False):
        # try to find an existing job processor for the job
        try:
            try:
                if force_reprocess:
                    raise FileNotFoundError  # if force reprocessing, pretend like we haven't found the job
                jp = JobProcessor.load(os.path.join(job_dir_path, '{}.job'.format(job_name)))
            except (AttributeError, TypeError):
                raise FileNotFoundError  # if something goes wrong, pretend like we haven't found it
        except (FileNotFoundError, EOFError, ImportError):
            job_info_path = os.path.join(job_dir_path, '{}_info.json'.format(job_name))
            with open(job_info_path, mode = 'r') as f:
                job_info = json.load(f)
            jp = job_info['job_processor'](job_name, job_dir_path)  # TODO: this won't work maybe?

        jp.process_job(individual_processing = individual_processing)  # try to detect if jp is from previous version and won't work, maybe catch attribute access exceptions?

    def process_jobs(self, individual_processing = False):
        start_time = dt.datetime.now()
        logger.info('Processing jobs')

        for job_name, job_dir_path in zip(self.job_dir_names, self.job_dir_paths):
            try:
                self.process_job(job_name, job_dir_path, individual_processing = individual_processing)
            except Exception as e:
                logger.exception(e)
                raise e

        end_time = dt.datetime.now()
        logger.info('Processing complete. Elapsed time: {}'.format(end_time - start_time))


class JobProcessor(utils.Beet):
    def __init__(self, job_name, job_dir_path, simulation_type):
        super(JobProcessor, self).__init__(job_name)
        self.job_dir_path = job_dir_path
        self.input_dir = os.path.join(self.job_dir_path, 'inputs')
        self.output_dir = os.path.join(self.job_dir_path, 'outputs')
        self.plots_dir = os.path.join(self.job_dir_path, 'plots')
        self.movies_dir = os.path.join(self.job_dir_path, 'movies')

        sim_names = [f.strip('.spec') for f in os.listdir(self.input_dir)]
        self.sim_names = sorted(sim_names, key = int)
        self.sim_count = len(self.sim_names)
        self.unprocessed_sims = set(self.sim_names)

        self.data = OrderedDict((sim_name, {}) for sim_name in self.sim_names)
        self.parameter_sets = defaultdict(set)

        self.simulation_type = simulation_type

    def __str__(self):
        return '{} for job {}, processed {}/{} Simulations'.format(self.__class__.__name__, self.name, self.sim_count - len(self.unprocessed_sims), self.sim_count)

    def save(self, target_dir = None, file_extension = '.job'):
        return super(JobProcessor, self).save(target_dir = target_dir, file_extension = file_extension)

    def collect_data_from_sim(self, sim_name, sim):
        """Hook method to collect summary data from a single Simulation."""
        self.data[sim_name].update({
            'name': sim.name,
            'file_name': int(sim.file_name),
            'start_time': sim.start_time,
            'end_time': sim.end_time,
            'elapsed_time': sim.elapsed_time.total_seconds(),
            'run_time': sim.run_time.total_seconds(),
        })

        for parameter, value in self.data[sim_name].items():
            try:
                self.parameter_sets[parameter].add(value)
            except TypeError as e:
                pass

    def process_sim(self, sim_name, sim):
        raise NotImplementedError

    def load_sim(self, sim_name, **load_kwargs):
        """
        Load a Simulation from the job by its file_name. load_kwargs are passed to the Simulation's load method

        :param sim_name:
        :param load_kwargs:
        :return: the loaded Simulation, or None if it wasn't found
        """
        sim = None

        try:
            sim = self.simulation_type.load(os.path.join(self.output_dir, '{}.sim'.format(sim_name)), **load_kwargs)

            if sim.status != 'finished':
                raise FileNotFoundError

            logger.debug('Loaded {}.sim from job {}'.format(sim_name, self.name))
        except (FileNotFoundError, EOFError) as e:
            logger.debug('Failed to find completed {}.sim from job {} due to {}'.format(sim_name, self.name, e))
        except Exception as e:
            logger.critical('Error while trying to find completed {}.sim from job {} due to {}'.format(sim_name, self.name, e))
            raise e

        return sim

    def process_job(self, individual_processing = False):
        start_time = dt.datetime.now()
        logger.info('Loading simulations from job {}'.format(self.name))

        for sim_name in tqdm(copy(self.unprocessed_sims)):
            sim = self.load_sim(sim_name)
            if sim is not None:
                self.collect_data_from_sim(sim_name, sim)
                if individual_processing:
                    self.process_sim(sim_name, sim)
                self.unprocessed_sims.remove(sim_name)

        end_time = dt.datetime.now()
        logger.info('Finished loading simulations from job {}. Failed to find {} / {} sims. Elapsed time: {}'.format(self.name, len(self.unprocessed_sims), self.sim_count, end_time - start_time))

    def write_to_csv(self):
        raise NotImplementedError

    def make_plot(self, name, x_key, *plot_lines, **kwargs):
        data = OrderedDict((k, v) for k, v in sorted(self.data.items(), key = lambda x: x[1][x_key] if x_key in x[1] else 0) if v)

        x_array = np.array(sorted(set(v[x_key] for v in data.values())))

        y_arrays = [np.array(list(v[plot_line.key] for v in data.values() if all(f(v) for f in plot_line.filters))) for plot_line in plot_lines]
        line_labels = (plot_line.label for plot_line in plot_lines)
        line_kwargs = (plot_line.line_kwargs for plot_line in plot_lines)

        utils.xy_plot(name, x_array, *y_arrays, line_labels = line_labels, line_kwargs = line_kwargs, **kwargs)


def check(parameter, value):
    def checker(v):
        return v[parameter] == value

    return checker


class KeyFilterLine:
    def __init__(self, key, filters = (lambda v: True,), label = None, **line_kwargs):
        self.key = key
        self.filters = filters
        if label is None:
            label = key
        self.label = label
        self.line_kwargs = line_kwargs

    def __str__(self):
        return 'Line: key = {}, filters = {}'.format(self.key, self.filters)

    def __repr__(self):
        return 'KeyFilterLine(key = {}, filter = {})'.format(self.key, self.filters)


class Parameter:
    name = utils.Typed('name', legal_type = str)
    expandable = utils.Typed('expandable', legal_type = bool)

    def __init__(self, name, value = None, expandable = False):
        self.name = name
        self.value = value
        self.expandable = expandable

    def __str__(self):
        return '{} {} = {}'.format(self.__class__.__name__, self.name, self.value)

    def __repr__(self):
        return '{}(name = {}, value = {})'.format(self.__class__.__name__, self.name, self.value)


def expand_parameters_to_dicts(parameters):
    dicts = [OrderedDict()]

    for par in parameters:
        if par.expandable and hasattr(par.value, '__iter__') and not isinstance(par.value, str) and hasattr(par.value, '__len__'):  # make sure the value is an iterable that isn't a string and has a length
            dicts = [deepcopy(d) for d in dicts for _ in range(len(par.value))]
            for d, v in zip(dicts, it.cycle(par.value)):
                d[par.name] = v
        else:
            for d in dicts:
                d[par.name] = par.value

    return dicts


def ask_for_input(question, default = None, cast_to = str):
    """Ask for input from the user, with a default value, and call cast_to on it before returning it."""
    try:
        input_str = input(question + ' [Default: {}] > '.format(default))

        trimmed = input_str.replace(' ', '')
        if trimmed == '':
            out = cast_to(default)
        else:
            out = cast_to(trimmed)

        logger.debug('Got input from stdin for question "{}": {}'.format(question, out))

        return out
    except Exception as e:
        print(e)
        ask_for_input(question, default = default, cast_to = cast_to)


def ask_for_bool(question, default = False):
    """

    Synonyms for True: 'true', 't', 'yes', 'y', '1', 'on'
    Synonyms for False: 'false', 'f', 'no', 'n', '0', 'off'
    :param question:
    :param default:
    :return:
    """
    try:
        input_str = input(question + ' [Default: {}] > '.format(default))

        trimmed = input_str.replace(' ', '')
        if trimmed == '':
            input_str = str(default)

        logger.debug('Got input from stdin for question "{}": {}'.format(question, input_str))

        input_str_lower = input_str.lower()
        if input_str_lower in ('true', 't', 'yes', 'y', '1', 'on'):
            return True
        elif input_str_lower in ('false', 'f', 'no', 'n', '0', 'off'):
            return False
        else:
            raise ValueError('Invalid answer to question "{}"'.format(question))
    except Exception as e:
        print(e)
        ask_for_bool(question, default = default)


def ask_for_eval(question, default = 'None'):
    input_str = input(question + ' [Default: {}] (eval) > '.format(default))

    trimmed = input_str.replace(' ', '')
    if trimmed == '':
        input_str = str(default)

    logger.debug('Got input from stdin for question "{}": {}'.format(question, input_str))

    try:
        return eval(input_str)
    except NameError as e:
        did_you_mean = ask_for_bool("Did you mean '{}'?".format(input_str), default = 'yes')
        if did_you_mean:
            return eval("'{}'".format(input_str))
        else:
            raise e
    except Exception as e:
        print(e)
        ask_for_eval(question, default = default)


def abort_job_creation():
    print('Aborting job creation...')
    logger.critical('Aborted job creation')
    sys.exit(0)


def create_job_dirs(job_dir):
    print('Creating job directory and subdirectories...')

    utils.ensure_dir_exists(job_dir)
    utils.ensure_dir_exists(os.path.join(job_dir, 'inputs'))
    utils.ensure_dir_exists(os.path.join(job_dir, 'outputs'))
    utils.ensure_dir_exists(os.path.join(job_dir, 'logs'))


def save_specifications(specifications, job_dir):
    print('Saving Parameters...')

    for spec in specifications:
        spec.save(target_dir = os.path.join(job_dir, 'inputs/'))

    logger.debug('Saved Specifications')


def write_specifications_info_to_file(specifications, job_dir):
    print('Writing Specification info to file...')

    with open(os.path.join(job_dir, 'specifications.txt'), 'w') as file:
        for spec in specifications:
            file.write(str(spec))
            file.write(spec.info())
            file.write('\n\n')  # blank line between specs

    logger.debug('Saved Specification information')


def write_parameters_info_to_file(parameters, job_dir):
    print('Writing Parameters info to file...')

    with open(os.path.join(job_dir, 'parameters.txt'), 'w') as file:
        for param in parameters:
            file.write(repr(param))
            file.write('\n')  # blank line between specs

    logger.debug('Saved Parameter information')


CHTC_SUBMIT_STRING = """universe = vanilla
log = logs/cluster_$(Cluster).log
error = logs/$(Process).err
#
executable = /home/karpel/backend/run_sim.sh
arguments = $(Process)
#
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = /home/karpel/backend/compy.tar.gz, /home/karpel/backend/ionization.tar.gz, /home/karpel/backend/run_sim.py, inputs/$(Process).spec, http://proxy.chtc.wisc.edu/SQUID/karpel/python.tar.gz
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


def specification_check(specifications):
    print('Generated {} Specifications'.format(len(specifications)))

    print('-' * 20)
    print(specifications[0])
    print(specifications[0].info())
    print('-' * 20)

    check = ask_for_bool('Does the first Specification look correct?', default = 'No')
    if not check:
        abort_job_creation()


def submit_check(submit_string):
    print('-' * 20)
    print(submit_string)
    print('-' * 20)

    check = ask_for_bool('Does the submit file look correct?', default = 'No')
    if not check:
        abort_job_creation()


def write_submit_file(submit_string, job_dir):
    print('Saving submit file...')

    with open(os.path.join(job_dir, 'submit_job.sub'), mode = 'w') as file:
        file.write(submit_string)

    logger.debug('Saved submit file')


def write_job_info(job_info, job_dir):
    with open(os.path.join(job_dir, 'info.json'), mode = 'w') as f:
        json.dump(job_info, f)


def submit_job(job_dir):
    print('Submitting job...')

    # TODO: temp chdir context manager
    os.chdir(job_dir)

    subprocess.run(['condor_submit', 'submit_job.sub'])
