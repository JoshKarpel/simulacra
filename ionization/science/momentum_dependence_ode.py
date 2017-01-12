import os
import logging
from copy import deepcopy

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        a_0 = 1

        pw = 200
        bound = 30

        times = np.linspace(-bound * pw * asec, bound * pw * asec, 1e5)
        dt = np.abs(times[1] - times[0])

        prefactor = ((electron_charge / hbar) ** 2) * np.sqrt(electron_mass / 2)

        electric_field = ion.SincPulse(pw * asec, 1 * Jcm2, phase = 'cos', dc_correction_time = bound * pw * asec)
        electric_field_amplitude_vs_time = electric_field.get_electric_field_amplitude(times)
        integral_of_electric_field_amplitude_vs_time = electric_field.get_total_electric_field_numeric(times)

        cp.utils.xy_plot('electric_field_vs_time',
                         times, electric_field_amplitude_vs_time,
                         x_scale = 'asec', y_scale = 'AEF',
                         target_dir = OUT_DIR)

        a = np.zeros(np.shape(times), dtype = np.complex128) * np.NaN
        a[0] = a_0
        print(a)


        def field_a_sum(through_index):
            return np.sum(a[:through_index] * electric_field_amplitude_vs_time[:through_index])


        print(a[:1])
        print(electric_field_amplitude_vs_time[:1])

        print(field_a_sum(0))
        print(field_a_sum(1))
        print(field_a_sum(2))

        print(prefactor * (dt ** 2) * field_a_sum(1))

        a[1] = a[0] + prefactor * (dt ** 2) * field_a_sum(1)
        print(a)
