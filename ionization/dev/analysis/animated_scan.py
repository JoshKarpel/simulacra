import os

import compy as cp
import compy.cluster as clu
from units import *
import ionization as ion
import ionization.cluster as iclu
import time

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.LogManager('compy', 'ionization'):
        jp = clu.JobProcessor.load('ide__sinc__25pw_20ph_20flu__v2.job')

        jp.plots_dir = OUT_DIR
        jp.summary_dir = OUT_DIR

        for pw in jp.parameter_set('pulse_width'):
            x = sorted(jp.parameter_set('phase'))
            fluences = sorted(jp.parameter_set('fluence'))


            def f(x, fluence):
                results = jp.select_by_kwargs(**{'fluence': fluence, 'pulse_width': pw})

                return [r.final_bound_state_overlap for r in results]


            cp.plots.xyt_plot(f'phase_fluence__pw={uround(pw, "asec", 3)}as',
                              x, fluences, f,
                              x_unit = 'rad', t_unit = 'Jcm2', t_fmt_string = r'$H = {}$',
                              y_log_axis = False,
                              length = 10,
                              target_dir = OUT_DIR)

            time.sleep(1)
