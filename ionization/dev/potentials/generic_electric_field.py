import logging
import os

import numpy as np

import compy as cp
import ionization as ion
import plots
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        pw = 200
        flu = 1

        bound = 30

        times = np.linspace(-pw * bound * asec, pw * bound * asec, 1e4)
        dt = np.abs(times[1] - times[0])
        sinc = ion.SincPulse(pw * asec, fluence = flu * Jcm2, phase = 'cos')

        print(sinc.amplitude_omega)
        print(sinc.amplitude_omega ** 2)

        print(2 * sinc.omega_max * epsilon_0 * c * sinc.amplitude_omega ** 2 / Jcm2)
        print(2 * sinc.frequency_max * epsilon_0 * c * (np.sqrt(twopi) * sinc.amplitude_omega) ** 2 / Jcm2)
        print(2 * sinc.frequency_max * epsilon_0 * c * (sinc.amplitude_per_frequency) ** 2 / Jcm2)

        logger.info('Sinc cutoff frequency: {} THz'.format(uround(sinc.frequency_max, THz)))

        generic = ion.GenericElectricField(lambda f: np.where(np.abs(f) < sinc.frequency_max, sinc.amplitude_per_frequency, 0),
                                           lambda f: np.where(f >= 0, pi / 2, -pi / 2),
                                           frequency_upper_limit = sinc.frequency_max * 20,
                                           frequency_points = 2 ** 15
                                           )

        for ii, phase in enumerate(np.arange(0, twopi + 0.01, pi / 8)):
            generic = ion.GenericElectricField(lambda f: np.where(np.abs(f) < sinc.frequency_max, sinc.amplitude_per_frequency, 0),
                                               lambda f: np.where(f >= 0, phase, -phase),
                                               frequency_upper_limit = sinc.frequency_max * 20,
                                               frequency_points = 2 ** 15
                                               )

            plots.xy_plot('generic_electric_field_vs_time__zoom__{}__phase={}'.format(ii, phase / pi),
                          generic.times, np.real(generic.complex_electric_field_vs_time),
                          x_scale = 'asec', x_label = r'Time $t$',
                          x_lower_limit = times[0], x_upper_limit = times[-1],
                          y_scale = 'atomic_electric_field', y_label = r'Electric Field $E(t)$',
                          target_dir = OUT_DIR)

            print(ii, phase, phase / pi)
            print('fluence', epsilon_0 * c * np.sum(np.abs(generic.complex_electric_field_vs_time) ** 2) * generic.dt / Jcm2)

            # print('dt', generic.dt / asec)
            # print('df', generic.df / THz)
            #
            # # print('expected center', sinc.amplitude_per_frequency)
            # # print('center', generic.complex_amplitude_vs_frequency[len(generic.complex_amplitude_vs_frequency) // 2])
            # # print(epsilon_0 * c * np.sum(generic.power_vs_frequency) * generic.df / Jcm2)
            #
            # cp.utils.xy_plot('power_spectrum',
            #                  generic.frequency, generic.power_vs_frequency,
            #                  x_scale = 'THz', x_label = r'Frequency $f$',
            #                  target_dir = OUT_DIR)
            #
            # cp.utils.xy_plot('generic_electric_field_vs_time',
            #                  generic.times, np.real(generic.complex_electric_field_vs_time),
            #                  x_scale = 'asec', x_label = r'Time $t$',
            #                  y_scale = 'atomic_electric_field', y_label = r'Electric Field $E(t)$',
            #                  target_dir = OUT_DIR)
            #
            # cp.utils.xy_plot('generic_electric_field_vs_time__zoom',
            #                  generic.times, np.real(generic.complex_electric_field_vs_time),
            #                  x_scale = 'asec', x_label = r'Time $t$',
            #                  x_lower_limit = times[0], x_upper_limit = times[-1],
            #                  y_scale = 'atomic_electric_field', y_label = r'Electric Field $E(t)$',
            #                  target_dir = OUT_DIR)
            #
            # sinc_field = sinc.get_electric_field_amplitude(times)
            # generic_field = generic.get_electric_field_amplitude(times)
            #
            # print(sinc_field[500])
            # print(generic_field[500])
            #
            # print(epsilon_0 * c * np.sum(np.abs(sinc_field) ** 2) * dt / Jcm2)
            # print(epsilon_0 * c * np.sum(np.abs(generic_field) ** 2) * dt / Jcm2)
            # print(epsilon_0 * c * np.sum(np.abs(generic.complex_electric_field_vs_time) ** 2) * generic.dt / Jcm2)
            #
            # cp.utils.xy_plot('electric_field_vs_time_comparison',
            #                  times,
            #                  sinc_field, generic_field,
            #                  line_labels = ('Sinc Pulse', 'Generic Pulse'),
            #                  x_scale = 'asec', x_label = r'Time $t$',
            #                  y_scale = 'atomic_electric_field', y_label = r'Electric Field $E(t)$',
            #                  target_dir = OUT_DIR)
