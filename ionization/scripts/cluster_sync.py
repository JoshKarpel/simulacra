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


def cluster_sync_loop(cluster_interface, wait_after_success = dt.timedelta(hours = 1), wait_after_failure = dt.timedelta(minutes = 1)):
    latest_sync_time = None
    while True:
        if latest_sync_time is None or dt.datetime.now() - latest_sync_time > wait_after_success:
            try:
                start_time = dt.datetime.now()
                logger.info('Beginning automatic synchronization')

                with cluster_interface as ci:
                    logger.info(ci.get_job_status())
                    ci.mirror_remote_home_dir()

                end_time = dt.datetime.now()
                logger.info('Synchronization complete. Elapsed time: {}'.format(end_time - start_time))

                latest_sync_time = end_time
                logger.info('Next automatic synchronization attempt after {}'.format(latest_sync_time + wait_after_success))
            except (FileNotFoundError, PermissionError, TimeoutError) as e:
                logger.exception('Exception encountered')
                logger.warning('Automatic synchronization attempt failed, retrying in {} seconds'.format(wait_after_failure.total_seconds()))

        time.sleep(wait_after_failure.total_seconds())


if __name__ == '__main__':
    with cp.utils.Logger('__main__', 'compy', 'ionization',
                         stdout_logs = True, stdout_level = logging.INFO,
                         file_logs = True, file_level = logging.WARNING, file_name = 'syncinc', file_dir = 'logs'):
        try:
            ci = clu.ClusterInterface('submit-5.chtc.wisc.edu', username = 'karpel', key_path = 'E:\chtc_ssh_private')
            cluster_sync_loop(ci)
        except Exception as e:
            logger.exception(e)
            raise e
