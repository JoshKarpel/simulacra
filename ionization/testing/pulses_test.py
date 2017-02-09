import os
import logging

import numpy as np

import compy as cp
from compy.units import *
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG):
        pw = 100
        flu = 1

        bound = 5

        OUT_DIR = os.path.join(OUT_DIR, 'pw={}as_flu={}Jcm2'.format(pw, flu))

        sinc_plain = ion.SincPulse(pulse_width = pw * asec, fluence = flu * Jcm2)
        sinc_with_carrier = ion.CarrierSincPulse(pulse_width = pw * asec, phase = 0)
        sech_with_carrier = ion.CarrierSechPulse(pulse_width = pw * asec, omega_carrier = sinc_with_carrier.omega_carrier)

        print('carrier f (THz):', sinc_with_carrier.omega_carrier / (twopi * THz))

        times = np.linspace(-bound * pw, bound * pw, bound * pw * 10) * asec

        print('fluences (J/cm^2):')
        print('plain sinc:', sinc_plain.get_fluence_numeric(times) / Jcm2)
        print('carrier sinc:', sinc_with_carrier.get_fluence_numeric(times) / Jcm2)
        print('carrier sech:', sech_with_carrier.get_fluence_numeric(times) / Jcm2)

        plot_kwargs = {
            'x_label': r'Time $t$',
            'x_scale': 'asec',
            'y_label': r'Electric Field $\mathcal{E}(t)$',
            'y_scale': 'AEF',
            'target_dir': OUT_DIR}

        cp.utils.xy_plot('sinc_pulse_comparison',
                         times,
                         sinc_plain.get_electric_field_amplitude(times),
                         sinc_with_carrier.get_electric_field_amplitude(times),
                         line_labels = ('No Carrier', 'w/ Carrier'),
                         **plot_kwargs)

        cp.utils.xy_plot('sinc_pulse_envelope',
                         times,
                         np.abs(sinc_plain.get_electric_field_amplitude(times)),
                         -np.abs(sinc_plain.get_electric_field_amplitude(times)),
                         sinc_with_carrier.get_electric_field_amplitude(times),
                         line_labels = ('Envelope', 'Envelope'),
                         line_kwargs = ({'color': 'red'}, {'color': 'red'}),
                         **plot_kwargs)

        cp.utils.xy_plot('sech_pulse',
                         times,
                         sech_with_carrier.get_electric_field_amplitude(times),
                         **plot_kwargs)
