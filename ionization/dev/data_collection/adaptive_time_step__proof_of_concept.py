import os
import logging

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        def get_time_step(current_time, electric_potential):
            abs_amp = np.abs(electric_potential.get_electric_field_amplitude(current_time))

            if abs_amp > 1 * atomic_electric_field:
                dt = .1 * asec
            elif abs_amp < .1 * atomic_electric_field and np.abs(current_time) > 5 * electric_potential.pulse_width:
                dt = 5 * asec
            else:
                dt = 1 * asec

            return dt


        # JUST PASS IN THE SPEC
        # REALISTICALLY, WILL PROBABLY JUST DO .1 asec BELOW 10 PW OR SOMETHING
        # REMOVE EXTRA TIME OPTIONS - THIS IS MORE FLEXIBLE

        # efield = ion.Rectangle(start_time = 50 * asec, end_time = 150 * asec, amplitude = .5 * atomic_electric_field) + ion.Rectangle(start_time = 100 * asec, end_time = 125 * asec, amplitude = 1 * atomic_electric_field)
        efield = ion.SincPulse(pulse_width = 100 * asec, fluence = 20 * Jcm2)

        t_init = -1000 * asec

        t_final = 1000 * asec

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

        steps = np.diff(times)
        steps = np.append(steps, np.NaN)

        cp.plots.xy_plot('test',
                         times,
                         steps / asec,
                         np.abs(efield.get_electric_field_amplitude(times)) / atomic_electric_field,
                         line_labels = (r'$\Delta t$', rf'${str_efield}(t)$'),
                         x_unit = 'asec',
                         target_dir = OUT_DIR,
                         )
