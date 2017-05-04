import logging
import os

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

import compy as cp
import ionization as ion
from compy.units import *


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
)

EA_FIELD_PLT_KWARGS = dict(
    line_labels = (fr'${str_efield}(t)$', fr'${str_afield}(t)$'),
    x_label = r'$t$',
    x_unit = 'asec',
    y_label = fr'${str_efield}(t)$, ${str_afield}(t)$',
)

EA_LOG_PLT_KWARGS = dict(
    y_log_axis = True,
    y_upper_limit = 2,
    y_lower_limit = 1e-20,
)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        pw = 100 * asec

        window = ion.SymmetricExponentialTimeWindow(window_time = 28 * pw, window_width = .2 * pw) + ion.RectangularTimeWindow(on_time = -31 * pw, off_time = 31 * pw)

        ref_sinc = ion.SincPulse(pulse_width = pw)
        print(ref_sinc)

        pulse = ion.SincPulse(pulse_width = pw, fluence = 1 * Jcm2, phase = pi / 2,
                              # omega_carrier = ref_sinc.omega_carrier,
                              window = window
                              )

        print(pulse)

        t = np.linspace(-35 * pw, 35 * pw, 1e6)
        total_t = np.abs(t[-1] - t[0])

        uncorrected_pulse_amp = pulse.get_electric_field_amplitude(t)
        uncorrected_pulse_vpot = proton_charge * (-pulse.get_electric_field_integral_numeric(t))

        print('uncorrected A final:', uncorrected_pulse_vpot[-1] / atomic_momentum)

        cp.plots.xy_plot(f'uncorrected_pulse',
                         t,
                         uncorrected_pulse_amp / atomic_electric_field,
                         uncorrected_pulse_vpot / atomic_momentum,
                         **EA_FIELD_PLT_KWARGS,
                         **PLT_KWARGS)

        cp.plots.xy_plot(f'uncorrected_pulse__log',
                         t,
                         np.abs(uncorrected_pulse_amp / atomic_electric_field),
                         np.abs(uncorrected_pulse_vpot / atomic_momentum),
                         **EA_FIELD_PLT_KWARGS,
                         **EA_LOG_PLT_KWARGS,
                         **PLT_KWARGS)

        ### CORRECTION 1 ###

        correction_field = ion.Rectangle(start_time = t[0], end_time = t[-1], amplitude = -pulse.get_electric_field_integral_numeric(t)[-1] / total_t)
        print(correction_field)

        corrected_pulse = pulse + correction_field
        print(corrected_pulse)

        corrected_pulse_amp = corrected_pulse.get_electric_field_amplitude(t)
        corrected_pulse_vpot = proton_charge * (-corrected_pulse.get_electric_field_integral_numeric(t))

        print('rect-corrected A final:', corrected_pulse_vpot[-1] / atomic_momentum)

        cp.plots.xy_plot(f'rect-corrected_pulse',
                         t,
                         corrected_pulse_amp / atomic_electric_field,
                         corrected_pulse_vpot / atomic_momentum,
                         **EA_FIELD_PLT_KWARGS,
                         **PLT_KWARGS)

        cp.plots.xy_plot(f'rect-corrected_pulse__log',
                         t,
                         np.abs(corrected_pulse_amp / atomic_electric_field),
                         np.abs(corrected_pulse_vpot / atomic_momentum),
                         **EA_FIELD_PLT_KWARGS,
                         **EA_LOG_PLT_KWARGS,
                         **PLT_KWARGS)


        ### CORRECTION 2 ###

        def func_to_minimize(amp, original_pulse):
            test_correction_field = ion.Rectangle(start_time = t[0], end_time = t[-1], amplitude = amp, window = original_pulse.window)
            test_pulse = original_pulse + test_correction_field

            return np.abs(test_pulse.get_electric_field_integral_numeric(t)[-1])


        correction_amp = optimize.minimize_scalar(func_to_minimize, args = (pulse,))

        print(correction_amp)

        correction_field = ion.Rectangle(start_time = t[0], end_time = t[-1], amplitude = correction_amp.x, window = pulse.window)
        print(correction_field)

        corrected_pulse = pulse + correction_field
        print(corrected_pulse)

        corrected_pulse_amp = corrected_pulse.get_electric_field_amplitude(t)
        corrected_pulse_vpot = proton_charge * (-corrected_pulse.get_electric_field_integral_numeric(t))

        print('opt-rect-corrected A final:', corrected_pulse_vpot[-1] / atomic_momentum)

        cp.plots.xy_plot(f'opt-rect-corrected_pulse',
                         t,
                         corrected_pulse_amp / atomic_electric_field,
                         corrected_pulse_vpot / atomic_momentum,
                         **EA_FIELD_PLT_KWARGS,
                         **PLT_KWARGS)

        cp.plots.xy_plot(f'opt-rect-corrected_pulse__log',
                         t,
                         np.abs(corrected_pulse_amp / atomic_electric_field),
                         np.abs(corrected_pulse_vpot / atomic_momentum),
                         **EA_FIELD_PLT_KWARGS,
                         **EA_LOG_PLT_KWARGS,
                         **PLT_KWARGS)
