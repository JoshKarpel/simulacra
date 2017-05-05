import os
import datetime as dt
import logging
import functools as ft

import compy as cp
import compy.cluster as clu
import ionization.cluster as iclu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_file = f"{__file__.strip('.py')}__{dt.datetime.now().strftime('%Y-%m-%d')}"
cp_logger = cp.utils.LogManager('__main__', 'compy', 'ionization',
                                stdout_logs = True, stdout_level = logging.INFO,
                                file_logs = False, file_level = logging.INFO, file_name = log_file, file_dir = os.path.join(os.getcwd(), 'logs'), file_mode = 'a')

DROPBOX_PROCESS_NAMES = ['Dropbox.exe']


def synchronize_with_cluster(cluster_interface):
    with cp.utils.SuspendProcesses(*DROPBOX_PROCESS_NAMES):
        with cluster_interface as ci:
            logger.info(ci.get_job_status())
            ci.mirror_remote_home_dir()


def process_job(job_name, jobs_dir = None):
    with cp_logger as l:
        if jobs_dir is None:
            jobs_dir = os.getcwd()
        job_dir = os.path.join(jobs_dir, job_name)

        job_info = clu.load_job_info_from_file(job_dir)

        try:
            jp = clu.JobProcessor.load(os.path.join(job_dir, job_name + '.job'))
            l.info('Loaded existing job processor for job {}'.format(job_name))
        except FileNotFoundError:
            jp = job_info['job_processor_type'](job_name, job_dir)
            l.info('Created new job processor for job {}'.format(job_name))

        with cp.utils.SuspendProcesses(*DROPBOX_PROCESS_NAMES):
            jp.load_sims(force_reprocess = False)

        jp.save(target_dir = os.path.join(os.getcwd(), 'job_processors'))

        jp.summarize()

        return jp.running_time, jp.sim_count


def process_jobs(jobs_dir):
    jobs_processed = 0
    total_runtime = dt.timedelta()
    total_sim_count = 0

    for job_name in (f for f in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, f))):
        try:
            logger.info('Found job {}'.format(job_name))
            runtime, sim_count = cp.utils.run_in_process(process_job, args = (job_name, jobs_dir))

            jobs_processed += 1
            total_runtime += runtime
            total_sim_count += sim_count
        except Exception as e:
            logger.exception('Encountered exception while processing job {}'.format(job_name))
            raise e

    logger.info(f'Processed {jobs_processed} jobs containing {total_sim_count} simulations, with total runtime {total_runtime}')


if __name__ == '__main__':
    with cp_logger as l:
        try:
            ci = clu.ClusterInterface('submit-5.chtc.wisc.edu', username = 'karpel', key_path = 'E:\chtc_ssh_private')
            jobs_dir = "E:\Dropbox\Research\Cluster\cluster_mirror\home\karpel\jobs"

            cp.utils.try_loop(
                ft.partial(synchronize_with_cluster, ci),
                ft.partial(process_jobs, jobs_dir),
                wait_after_success = dt.timedelta(hours = 3),
                wait_after_failure = dt.timedelta(hours = 1),
            )
        except Exception as e:
            logger.exception(e)
            raise e
