import os
import sys
import time
import datetime as dt
import logging
import argparse
import functools as ft
import psutil
from pprint import pprint

import compy as cp
import ionization.cluster as clu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def synchronize_with_cluster(cluster_interface):
    with cluster_interface as ci:
        logger.info(ci.get_job_status())
        ci.mirror_remote_home_dir()


def process_job(job_name, jobs_dir = None):
    if jobs_dir is None:
        jobs_dir = os.getcwd()
    job_dir = os.path.join(jobs_dir, job_name)

    job_info = clu.load_job_info_from_file(job_dir)

    try:
        jp = clu.JobProcessor.load(os.path.join(job_dir, job_name + '.job'))
        logger.info('Loaded existing job processor for job {}'.format(job_name))
    except FileNotFoundError:
        jp = job_info['job_processor_type'](job_name, job_dir)
        logger.info('Created new job processor for job {}'.format(job_name))

    jp.process_job(force_reprocess = False)

    jp.save(target_dir = job_dir)


def process_jobs(jobs_dir):
    for job_name in (f for f in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, f))):
        logger.info('Found job {}'.format(job_name))
        process_job(job_name, jobs_dir = jobs_dir)


def suspend_process(process):
    process.suspend()
    logger.info('Suspended {}'.format(process))


def resume_process(process):
    process.resume()
    logger.info('Resumed {}'.format(process))


if __name__ == '__main__':
    with cp.utils.Logger('__main__', 'compy', 'ionization',
                         stdout_logs = True, stdout_level = logging.INFO,
                         file_logs = True, file_level = logging.WARNING, file_name = '{}_{}'.format(__file__, dt.date.today().strftime('%Y-%m-%d_%H_%M_%S')), file_dir = 'logs'):
        dropbox_process = cp.utils.get_process_by_name('Dropbox.exe')

        try:
            ci = clu.ClusterInterface('submit-5.chtc.wisc.edu', username = 'karpel', key_path = 'E:\chtc_ssh_private')
            jobs_dir = "E:\Dropbox\Research\Cluster\cluster_mirror\home\karpel\jobs"

            cp.utils.try_loop(
                # ft.partial(suspend_process, dropbox_process),
                ft.partial(synchronize_with_cluster, ci),
                ft.partial(process_jobs, jobs_dir),
                # ft.partial(resume_process, dropbox_process),
                wait_after_success = dt.timedelta(hours = 3),
                wait_after_failure = dt.timedelta(hours = 1)
            )
        except Exception as e:
            logger.exception(e)
            raise e
        finally:
            resume_process(dropbox_process)
