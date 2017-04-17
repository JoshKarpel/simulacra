import os
import logging
from copy import deepcopy

import numpy as np
# import matplotlib.pyplot as plt

import compy as cp
import ionization as ion
import plots
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        fluence = 1

        bound = 20

        for pulse_width in (50, 100, 200, 500, 1000):
            times = np.linspace(-bound * pulse_width * asec, bound * pulse_width * asec, 1e4)

            window = ion.SymmetricExponentialTimeWindow(window_time = bound * pulse_width * asec, window_width = pulse_width * asec / 2)

            efield_kwargs = dict(
                pulse_width = pulse_width * asec,
                fluence = fluence * Jcm2,
            )

            omega_c = twopi * 20000 * THz

            # sinc_cos = ion.SincPulse(**efield_kwargs, phase = 0)
            # sinc_sin = ion.SincPulse(**efield_kwargs, phase = pi / 2)
            #
            # gaus_cos = ion.GaussianPulse(**efield_kwargs, phase = 0, omega_carrier = 2 * sinc_cos.omega_carrier)
            # gaus_sin = ion.GaussianPulse(**efield_kwargs, phase = pi / 2, omega_carrier = 2 * sinc_cos.omega_carrier)
            #
            # sech_cos = ion.SechPulse(**efield_kwargs, phase = 0, omega_carrier = 2 * sinc_cos.omega_carrier)
            # sech_sin = ion.SechPulse(**efield_kwargs, phase = pi / 2, omega_carrier = 2 * sinc_cos.omega_carrier)

            sinc_cos = ion.SincPulse.from_omega_carrier(**efield_kwargs, phase = 0, omega_carrier = omega_c)
            sinc_sin = ion.SincPulse.from_omega_carrier(**efield_kwargs, phase = pi / 2, omega_carrier = omega_c)

            print('delta', sinc_cos.delta_omega / (twopi * THz))
            print('min', sinc_cos.omega_min / (twopi * THz))
            print('carrier', sinc_cos.omega_carrier / (twopi * THz))
            print('max', sinc_cos.omega_max / (twopi * THz))
            print()

            gaus_cos = ion.GaussianPulse(**efield_kwargs, phase = 0, omega_carrier = omega_c)
            gaus_sin = ion.GaussianPulse(**efield_kwargs, phase = pi / 2, omega_carrier = omega_c)

            sech_cos = ion.SechPulse(**efield_kwargs, phase = 0, omega_carrier = omega_c)
            sech_sin = ion.SechPulse(**efield_kwargs, phase = pi / 2, omega_carrier = omega_c)

            field_prefactor = electron_charge  # convert to momentum

            plots.xy_plot('pw={}as_flu={}Jcm2_field'.format(pulse_width, fluence),
                          times,
                          sinc_cos.get_electric_field_amplitude(times),
                          sinc_sin.get_electric_field_amplitude(times),
                          gaus_cos.get_electric_field_amplitude(times),
                          gaus_sin.get_electric_field_amplitude(times),
                          sech_cos.get_electric_field_amplitude(times),
                          sech_sin.get_electric_field_amplitude(times),
                          line_labels = (r'Sinc $\varphi = 0$',
                                            r'Sinc $\varphi = \pi/2$',
                                            r'Gaussian $\varphi = 0$',
                                            r'Gaussian $\varphi = \pi/2$',
                                            r'Sech $\varphi = 0$',
                                            r'Sech $\varphi = \pi/2$',
                                            ),
                          line_kwargs = (
                                 dict(color = 'C0', linestyle = '-', linewidth = 0.5),
                                 dict(color = 'C0', linestyle = '--', linewidth = 0.5),
                                 dict(color = 'C1', linestyle = '-', linewidth = 0.5),
                                 dict(color = 'C1', linestyle = '--', linewidth = 0.5),
                                 dict(color = 'C2', linestyle = '-', linewidth = 0.5),
                                 dict(color = 'C2', linestyle = '--', linewidth = 0.5),
                             ),
                          x_scale = 'asec', y_scale = 'atomic_electric_field',
                          x_label = r'Time $t$', y_label = r'Electric Field $E(t)$',
                          target_dir = OUT_DIR)

            plots.xy_plot('pw={}as_flu={}Jcm2_integrated'.format(pulse_width, fluence),
                          times,
                          sinc_cos.get_total_electric_field_numeric(times) * field_prefactor,
                          sinc_sin.get_total_electric_field_numeric(times) * field_prefactor,
                          gaus_cos.get_total_electric_field_numeric(times) * field_prefactor,
                          gaus_sin.get_total_electric_field_numeric(times) * field_prefactor,
                          sech_cos.get_total_electric_field_numeric(times) * field_prefactor,
                          sech_sin.get_total_electric_field_numeric(times) * field_prefactor,
                          line_labels = (r'Sinc $\varphi = 0$',
                                            r'Sinc $\varphi = \pi/2$',
                                            r'Gaussian $\varphi = 0$',
                                            r'Gaussian $\varphi = \pi/2$',
                                            r'Sech $\varphi = 0$',
                                            r'Sech $\varphi = \pi/2$',
                                            ),
                          line_kwargs = (
                                 dict(color = 'C0', linestyle = '-', linewidth = 0.5),
                                 dict(color = 'C0', linestyle = '--', linewidth = 0.5),
                                 dict(color = 'C1', linestyle = '-', linewidth = 0.5),
                                 dict(color = 'C1', linestyle = '--', linewidth = 0.5),
                                 dict(color = 'C2', linestyle = '-', linewidth = 0.5),
                                 dict(color = 'C2', linestyle = '--', linewidth = 0.5),
                             ),
                          x_scale = 'asec', y_scale = 'atomic_momentum',
                          x_label = r'Time $t$', y_label = r'$e \, \int_{-\infty}^{t} E(\tau) \, \mathrm{d}\tau$',
                          target_dir = OUT_DIR)
