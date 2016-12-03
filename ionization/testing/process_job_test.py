import os
import logging
from pprint import pprint

import compy as cp
from compy.units import *
import ionization as ion
import ionization.cluster as clu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.WARNING):
        # job_name = 'both_phases'
        job_name = 'both_phases__multi_fluence__min_time'
        job_dir = os.path.join(OUT_DIR, job_name)

        try:
            jp = clu.SincPulseJobProcessor.load(os.path.join(job_dir, job_name + '.job'))
        except FileNotFoundError:
            jp = clu.SincPulseJobProcessor(job_name, job_dir)

        jp.process_job(individual_processing = False)

        jp.save(target_dir = job_dir)

        print(jp)
        with open(os.path.join(job_dir, 'data.txt'), mode = 'w') as f:
            pprint(jp.data, stream = f)

        jp.make_plot('pulse_width', 'run_time', 'elapsed_time',
                     name = 'diag', target_dir = job_dir,
                     labels = ('Run Time', 'Elapsed Time'),
                     y_scale = 'hours', y_label = 'Time to Complete',
                     x_scale = 'asec', x_label = 'Simulation Pulse Width')

        # jp.make_plot('pulse_width', 'final_norm', filter = lambda v: v['phase'] == 'cos',
        #              name = 'cos', target_dir = job_dir,
        #              y_label = 'Norm',
        #              x_scale = 'asec', x_label = 'Pulse Width')
        # jp.make_plot('pulse_width', 'final_norm', filter = lambda v: v['phase'] == 'sin',
        #              name = 'sin', target_dir = job_dir,
        #              y_label = 'Norm',
        #              x_scale = 'asec', x_label = 'Pulse Width')

        jp.make_plot('pulse_width', 'final_initial_state_overlap', filter = lambda v: v['phase'] == 'cos',
                     name = 'cos', target_dir = job_dir,
                     y_label = 'Final Initial State Overlap',
                     x_scale = 'asec', x_label = 'Pulse Width')
        jp.make_plot('pulse_width', 'final_initial_state_overlap', filter = lambda v: v['phase'] == 'sin',
                     name = 'sin', target_dir = job_dir,
                     y_label = 'Final Initial State Overlap',
                     x_scale = 'asec', x_label = 'Pulse Width')

        jp.make_plot('pulse_width', 'final_initial_state_overlap', filter = lambda v: v['phase'] == 'cos',
                     name = 'cos_log', target_dir = job_dir,
                     y_label = 'Final Initial State Overlap', log_y = True,
                     x_scale = 'asec', x_label = 'Pulse Width')
        jp.make_plot('pulse_width', 'final_initial_state_overlap', filter = lambda v: v['phase'] == 'sin',
                     name = 'sin_log', target_dir = job_dir,
                     y_label = 'Final Initial State Overlap', log_y = True,
                     x_scale = 'asec', x_label = 'Pulse Width')

        jp.make_plot('pulse_width', 'final_initial_state_overlap', filter = (lambda v: v['phase'] == 'cos', lambda v: v['phase'] == 'sin'),
                     legends = ('Cos', 'Sin'),
                     name = 'both_log', target_dir = job_dir,
                     y_label = 'Final Initial State Overlap', log_y = True,
                     x_scale = 'asec', x_label = 'Pulse Width')