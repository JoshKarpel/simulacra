import os
import argparse
import subprocess

import compy as cp
import compy.cluster as clu


def get_submit_file_paths(*job_names, jobs_dir = None):
    if jobs_dir is None:
        jobs_dir = os.getcwd()

    return tuple(os.path.abspath(os.path.join(jobs_dir, job_dir, 'submit_job.sub')) for job_dir in job_names)


def write_dag_files(name, job_names, submit_file_paths, max_jobs_idle = 5000, overwrite = False):
    dag_name = name + '.dag'
    con_name = name + '.config'

    dag_str = [f'CONFIG {con_name}']
    dag_str += [f'JOB {job_name} {submit_file_path}' for job_name, submit_file_path in zip(job_names, submit_file_paths)]

    if os.path.exists(dag_name):
        if not overwrite and not clu.ask_for_bool('A DAG with that name already exists. Overwrite?', default = 'No'):
            clu.abort_job_creation()
        else:
            os.remove(dag_name)
            os.remove(con_name)

    with open(dag_name, mode = 'w', newline = '') as f:
        print('\n'.join(dag_str))
        f.write('\n'.join(dag_str))

    with open(con_name, mode = 'w', newline = '') as f:
        f.write(f'DAGMAN_MAX_JOBS_IDLE = {max_jobs_idle}')

    return dag_name


def submit_dag(dag_file_name):
    print('Submitting DAG...')

    subprocess.run(['condor_submit_dag', dag_file_name])


if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser(description = 'Create DAGMan manager and config files.')
    parser.add_argument('dag_name',
                        type = str,
                        help = 'the name of the DAG job')
    parser.add_argument('job_names', metavar = 'Jobs', type = str, nargs = '+',
                        help = 'job names, space-separated')
    parser.add_argument('--max_jobs', '-m',
                        action = 'store', type = int, default = 5000,
                        help = 'max number of jobs for DAG to keep idle')
    parser.add_argument('--dir', '-d',
                        action = 'store', default = os.getcwd(),
                        help = 'directory to put the DAG files in. Defaults to cwd')
    parser.add_argument('--overwrite', '-o',
                        action = 'store_true',
                        help = 'force overwrite existing DAG files if there is a name collision')
    parser.add_argument('--verbosity', '-v',
                        action = 'count', default = 0,
                        help = 'set verbosity level')
    parser.add_argument('--dry',
                        action = 'store_true',
                        help = 'do not attempt to actually submit the job')

    args = parser.parse_args()

    with cp.utils.LogManager('compy', 'ionization', stdout_level = 31 - ((args.verbosity + 1) * 10)) as logger:
        submit_file_paths = get_submit_file_paths(*args.job_names)

        dag_file_name = write_dag_files(args.dag_name, args.job_names, submit_file_paths, max_jobs_idle = args.max_jobs, overwrite = args.overwrite)

        if not args.dry:
            submit_dag(dag_file_name)
