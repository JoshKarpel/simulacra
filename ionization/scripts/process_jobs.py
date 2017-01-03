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

JOBS_DIR = os.getcwd()


def process_job(job_name, jobs_dir = None):
    if jobs_dir is None:
        jobs_dir = os.getcwd()
    job_dir = os.path.join(jobs_dir, job_name)

    job_processor = clu.load_job_info_from_file(job_dir)['job_processor']

    try:
        jp = job_processor.load(os.path.join(job_dir, job_name + '.job'))
    except FileNotFoundError:
        jp = job_processor(job_name, job_dir)

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

                for job_name in (f for f in os.listdir(jobs_dir) if os.path.isdir(f)):
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
