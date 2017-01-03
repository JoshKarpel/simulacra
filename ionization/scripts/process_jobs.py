import os
import sys
import time
import datetime as dt
import logging
import argparse
from pprint import pprint

import compy as cp
import ionization.cluster as clu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JOBS_DIR = "E:\Dropbox\Research\Cluster\cluster_mirror\home\karpel\jobs"


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


def process_jobs_loop(jobs_dir, wait_after_success = dt.timedelta(hours = 1), wait_after_failure = dt.timedelta(minutes = 1)):
    latest_sync_time = None
    while True:
        if latest_sync_time is None or dt.datetime.now() - latest_sync_time > wait_after_success:
            try:
                start_time = dt.datetime.now()
                logger.info('Beginning automatic processing')

                for job_name in (f for f in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, f))):
                    logger.info('Found job {}'.format(job_name))
                    process_job(job_name, jobs_dir = jobs_dir)

                end_time = dt.datetime.now()
                logger.info('Processing complete. Elapsed time: {}'.format(end_time - start_time))

                latest_sync_time = end_time
                logger.info('Next automatic processing attempt after {}'.format(latest_sync_time + wait_after_success))
            except (FileNotFoundError, PermissionError, TimeoutError) as e:
                logger.exception('Exception encountered')
                logger.warning('Automatic processing attempt failed, retrying in {} seconds'.format(wait_after_failure.total_seconds()))

        time.sleep(wait_after_failure.total_seconds())


if __name__ == '__main__':
    with cp.utils.Logger('__main__', 'compy', 'ionization',
                         stdout_logs = True, stdout_level = logging.INFO,
                         file_logs = True, file_level = logging.WARNING, file_name = 'processing', file_dir = 'logs'):
        try:
            process_jobs_loop(JOBS_DIR)
        except Exception as e:
            logger.exception(e)
            raise e
