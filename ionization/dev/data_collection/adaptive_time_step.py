import logging
import os

import compy as cp
import numpy as np

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        def get_time_step(current_time, electric_potential):
            abs_amp = np.abs(electric_potential.get_electric_field_amplitude(current_time))

            if abs_amp > 1:
                return .1
            elif abs_amp > .1:
                return 1
            else:
                return 5


        efield = ion.Rectangle(start_time = 50, end_time = 150, amplitude = .5) + ion.Rectangle(start_time = 100, end_time = 125, amplitude = 1)

        t_init = 0

        t_final = 5e6

        t = t_init
        times = [t]

        with cp.utils.BlockTimer() as timer:
            while t < t_final:
                t += get_time_step(t, efield)

                if t > t_final:
                    t = t_final

                times.append(t)

        times = np.array(times)

        print(timer)
        print(len(times), times)

        cp.plots.xy_plot('')
