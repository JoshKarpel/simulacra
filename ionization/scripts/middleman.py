import os
import sys
import time
import datetime as dt
import logging
import argparse

import compy as cp
import compy.cluster as clu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def sync_process_loop(self, wait_after_success = dt.timedelta(hours = 1), wait_after_failure = dt.timedelta(minutes = 1)):
    latest_sync_time = None
    while True:
        if latest_sync_time is None or dt.datetime.now() - latest_sync_time > wait_after_success:
            try:
                start_time = dt.datetime.now()
                logger.info('Beginning automatic synchronization and processing')

                logger.info(self.job_status)

                # TODO: sync and process

                end_time = dt.datetime.now()
                logger.info('Synchronization and processing complete. Elapsed time: {}'.format(end_time - start_time))

                latest_sync_time = end_time
                logger.info('Next automatic synchronization attempt after {}'.format(latest_sync_time + wait_after_success))
            except (FileNotFoundError, PermissionError, TimeoutError) as e:
                logger.exception('Exception encountered')
                logger.warning('Automatic synchronization attempt failed')

        time.sleep(wait_after_failure.total_seconds())


if __name__ == '__main__':
    with cp.utils.Logger('__main__', 'compy', 'ionization',
                         stdout_logs = True, stdout_level = logging.INFO,
                         file_logs = True, file_level = logging.WARNING):
        pass
