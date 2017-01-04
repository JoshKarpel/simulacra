import os
import sys
import time
import datetime as dt
import logging
import argparse
import functools as ft
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
    if len(os.listdir(os.path.join(job_dir, 'inputs'))) != job_info['number_of_sims']:
        logger.info('Job {} has not finished synchronizing specifications, aborting'.format(job_name))
        return  # don't both processing the job if we haven't seen all the inputs yet

    job_processor = job_info['job_processor']

    try:
        jp = job_processor.load(os.path.join(job_dir, job_name + '.job'))
        logger.info('Loaded existing job processor for job {}'.format(job_name))
    except FileNotFoundError:
        jp = job_processor(job_name, job_dir)
        logger.info('Created new job processor for job {}'.format(job_name))

    jp.process_job(individual_processing = False, force_reprocess = True)

    jp.save(target_dir = job_dir)

    with open(os.path.join(job_dir, 'data.txt'), mode = 'w') as f:
        pprint(jp.data, stream = f)


def process_jobs(jobs_dir):
    for job_name in (f for f in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, f))):
        logger.info('Found job {}'.format(job_name))
        process_job(job_name, jobs_dir = jobs_dir)




if __name__ == '__main__':
    with cp.utils.Logger('__main__', 'compy', 'ionization',
                         stdout_logs = True, stdout_level = logging.INFO,
                         file_logs = True, file_level = logging.WARNING, file_name = 'syncinc', file_dir = 'logs'):
        try:
            CI = clu.ClusterInterface('submit-5.chtc.wisc.edu', username = 'karpel', key_path = 'E:\chtc_ssh_private')
            JOBS_DIR = "E:\Dropbox\Research\Cluster\cluster_mirror\home\karpel\jobs"

            cp.utils.try_loop(ft.partial(synchronize_with_cluster, CI),
                              ft.partial(process_jobs, JOBS_DIR))
        except Exception as e:
            logger.exception(e)
            raise e
