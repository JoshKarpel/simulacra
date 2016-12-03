import os
import sys
import time
import datetime as dt
import logging
import argparse

import compy as cp
import ionization.cluster as clu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def job_process_loop(jobs_dir, wait_after_success = dt.timedelta(hours = 1), wait_after_failure = dt.timedelta(minutes = 1)):
    latest_process_time = None
    while True:
        if latest_process_time is None or dt.datetime.now() - latest_process_time > wait_after_success:
            try:
                start_time = dt.datetime.now()
                logger.info('Beginning automatic synchronization and processing')

                try:
                    for job_name in os.listdir(jobs_dir):
                        job_dir = os.path.join(jobs_dir, job_name)
                        try:
                            print(job_name, job_dir)
                            jp = clu.SincPulseJobProcessor.load(os.path.join(job_dir, job_name + '.job'))
                        except FileNotFoundError:
                            jp = clu.SincPulseJobProcessor(job_name, job_dir)

                        jp.process_job(individual_processing = False)

                        jp.save(target_dir = job_dir)
                except Exception as e:
                    logger.exception(e)

                end_time = dt.datetime.now()
                logger.info('Synchronization and processing complete. Elapsed time: {}'.format(end_time - start_time))

                latest_process_time = end_time
                logger.info('Next automatic synchronization attempt after {}'.format(latest_process_time + wait_after_success))
            except (FileNotFoundError, PermissionError, TimeoutError) as e:
                logger.exception('Exception encountered')
                logger.warning('Automatic synchronization attempt failed, retrying in {} seconds'.format(wait_after_failure.total_seconds()))

        time.sleep(wait_after_failure.total_seconds())


if __name__ == '__main__':
    with cp.utils.Logger('__main__', 'compy', 'ionization',
                         stdout_logs = True, stdout_level = logging.DEBUG,
                         file_logs = True, file_level = logging.WARNING, file_dir = 'logs'):
        job_dir = 'D:\\GitHubProjects\\compy\\ionization\\testing\\out\\process_job_test'

        job_process_loop(job_dir)
