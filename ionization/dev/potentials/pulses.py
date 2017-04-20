import os
import logging

import numpy as np

import compy as cp
import plots
from compy.units import *
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG):
        pw = 200
        flu = 1
        phase = pi / 2

        bound = 10

        prefix = 'pw={}as_flu={}Jcm2'.format(pw, flu)

        pulse_kwargs = dict(
            pulse_width = pw * asec,
            fluence = flu * Jcm2,
            phase = phase,
        )

        # omega_c = twopi * 50000 * THz

        sinc = ion.SincPulse(**pulse_kwargs, omega_min = twopi * 1000 * THz)
        # sinc = ion.SincPulse.from_omega_carrier(**pulse_kwargs, omega_carrier = omega_c)
        # gaus = ion.GaussianPulse(**pulse_kwargs, omega_carrier = omega_c)
        gaus = ion.GaussianPulse(**pulse_kwargs, omega_carrier = sinc.omega_carrier)
        sech = ion.SechPulse(**pulse_kwargs, omega_carrier = sinc.omega_carrier)
        # sech = ion.SechPulse(**pulse_kwargs, omega_carrier = omega_c)

        print('carrier f (THz):', )
        print(sinc.omega_carrier / (twopi * THz))
        print(gaus.omega_carrier / (twopi * THz))
        print(sech.omega_carrier / (twopi * THz))

        times = np.linspace(-bound * pw, bound * pw, bound * pw * 50) * asec

        print('fluences (J/cm^2):')
        print('sinc:', sinc.get_fluence_numeric(times) / Jcm2)
        print('gaus:', gaus.get_fluence_numeric(times) / Jcm2)
        print('sech:', sech.get_fluence_numeric(times) / Jcm2)

        plot_kwargs = {
            'x_label': r'Time $t$',
            'x_unit': 'asec',
            'y_label': r'Electric Field $\mathcal{E}(t)$',
            'y_unit': 'AEF',
            'target_dir': OUT_DIR}

        plots.xy_plot(prefix + '__pulse_comparison',
                      times,
                      sinc.get_electric_field_amplitude(times),
                      gaus.get_electric_field_amplitude(times),
                      sech.get_electric_field_amplitude(times),
                      line_labels = ('Sinc', 'Gaussian', 'Sech'),
                      **plot_kwargs
                      )


        # cp.utils.xy_plot('sinc_pulse_comparison',
        #                  times,
        #                  sinc_plain.get_electric_field_amplitude(times),
        #                  sinc.get_electric_field_amplitude(times),
        #                  line_labels = ('No Carrier', 'w/ Carrier'),
        #                  **plot_kwargs)
        #
        # cp.utils.xy_plot('sinc_pulse_envelope',
        #                  times,
        #                  np.abs(sinc_plain.get_electric_field_amplitude(times)),
        #                  -np.abs(sinc_plain.get_electric_field_amplitude(times)),
        #                  sinc.get_electric_field_amplitude(times),
        #                  line_labels = ('Envelope', 'Envelope'),
        #                  line_kwargs = ({'color': 'red'}, {'color': 'red'}),
        #                  **plot_kwargs)
        #
        # cp.utils.xy_plot('sech_pulse',
        #                  times,
        #                  sech_with_carrier.get_electric_field_amplitude(times),
        #                  **plot_kwargs)
