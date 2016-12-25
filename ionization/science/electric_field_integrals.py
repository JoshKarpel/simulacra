import os
import logging
from copy import deepcopy

import numpy as np
# import matplotlib.pyplot as plt

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        pulse_width = 200
        fluence = 1

        bound = 25

        times = np.linspace(-bound * pulse_width * asec, bound * pulse_width * asec, 1e4)

        sinc_cos = ion.SincPulse(pulse_width = pulse_width * asec, fluence = fluence * J / (cm ** 2), phase = 'cos',
                                 dc_correction_time = bound * pulse_width * asec)
        sinc_sin = ion.SincPulse(pulse_width = pulse_width * asec, fluence = fluence * J / (cm ** 2), phase = 'sin')

        field_prefactor = electron_charge  # convert to momentum

        cp.utils.xy_plot('pw={}as_flu={}Jcm2_field'.format(pulse_width, fluence), times,
                         sinc_cos.get_amplitude(times), sinc_sin.get_amplitude(times),
                         line_labels = ('Cos', 'Sin'),
                         x_scale = 'asec', y_scale = 'atomic_electric_field',
                         x_label = r'Time $t$', y_label = r'Electric Field $E(t)$',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('pw={}as_flu={}Jcm2_integrated'.format(pulse_width, fluence), times,
                         sinc_cos.get_total_electric_field_numeric(times) * field_prefactor,
                         sinc_sin.get_total_electric_field_numeric(times) * field_prefactor,
                         line_labels = ('Cos', 'Sin'),
                         x_scale = 'asec', y_scale = atomic_momentum,
                         x_label = r'Time $t$', y_label = r'$e \, \int^{t} E(\tau) \, \mathrm{d}\tau$',
                         target_dir = OUT_DIR)
