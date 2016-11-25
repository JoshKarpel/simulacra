import matplotlib

matplotlib.use('Agg')

import os
import logging

import compy as cp
from compy.units import *
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = False, stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR) as logger:
        animators = [
            ion.animators.SphericalHarmonicAnimator(target_dir = OUT_DIR, postfix = 'full'),
            ion.animators.SphericalHarmonicAnimator(target_dir = OUT_DIR, postfix = '30', plot_limit = 30 * bohr_radius),
            ion.animators.SphericalHarmonicAnimator(target_dir = OUT_DIR, postfix = '30_no_renorm', plot_limit = 30 * bohr_radius, renormalize_l_decomposition = False)
        ]

        e_pot = ion.potentials.Rectangle(amplitude = 3 * atomic_electric_field, window = ion.potentials.LinearRampWindow(ramp_on_time = 10 * asec, ramp_time = 10 * asec))

        sim = ion.SphericalHarmonicSpecification('test',
                                                 time_initial = 0 * asec, time_final = 50 * asec, time_step = 1 * asec,
                                                 r_bound = 50 * bohr_radius, r_points = 200, l_points = 30,
                                                 electric_potential = e_pot,
                                                 animators = animators
                                                 ).to_simulation()

        sim.run_simulation()
        logger.info(sim.info())
        print(sim.info())
