import matplotlib

matplotlib.use('Agg')

import os
import logging

import compy as cp
from compy.units import *
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def make_movie(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        sim = ion.ElectricFieldSimulation(spec)

        sim.run_simulation()

        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        initial_state = ion.BoundState(1, 0, 0)

        bound = 100
        points = 2 ** 10
        angular_points = 2 ** 5
        dt = 2.5 * asec

        period = 100 * asec

        t_init = -8 * period
        t_final = -t_init

        window = ion.potentials.LinearRampWindow(ramp_on_time = t_init + period, ramp_time = 4 * period)
        e_field = ion.potentials.SineWave(omega = twopi / period, amplitude = 2 * atomic_electric_field, window = window)

        specs = []

        # animators = [ion.animators.CylindricalSliceAnimator(target_dir = OUT_DIR), ion.animators.CylindricalSliceAnimator(postfix = 'log', target_dir = OUT_DIR, log = True)]
        animators = [ion.animators.CylindricalSliceAnimator(target_dir = OUT_DIR)]
        spec = ion.CylindricalSliceSpecification('cyl_slice', time_initial = t_init, time_final = t_final, time_step = dt,
                                                 z_bound = bound * bohr_radius, z_points = points,
                                                 rho_bound = bound * bohr_radius, rho_points = points / 2,
                                                 initial_state = initial_state,
                                                 electric_potential = e_field,
                                                 animators = animators
                                                 )
        specs.append(spec)

        # animators = [ion.animators.SphericalSliceAnimator(target_dir = OUT_DIR), ion.animators.SphericalSliceAnimator(postfix = 'log', target_dir = OUT_DIR, log = True)]
        animators = [ion.animators.SphericalSliceAnimator(target_dir = OUT_DIR)]
        spec = ion.SphericalSliceSpecification('sph_slice', time_initial = t_init, time_final = t_final, time_step = dt,
                                               r_bound = bound * bohr_radius, r_points = points,
                                               theta_points = angular_points,
                                               initial_state = initial_state,
                                               electric_potential = e_field,
                                               animators = animators
                                               )
        specs.append(spec)

        # animators = [ion.animators.SphericalSliceAnimator(target_dir = OUT_DIR), ion.animators.SphericalSliceAnimator(postfix = 'log', target_dir = OUT_DIR, log = True)]
        animators = [ion.animators.SphericalHarmonicAnimator(target_dir = OUT_DIR)]
        spec = ion.SphericalHarmonicSpecification('sph_harm', time_initial = t_init, time_final = t_final, time_step = dt,
                                                  r_bound = bound * bohr_radius, r_points = points,
                                                  spherical_harmonic_max_l = angular_points,
                                                  initial_state = initial_state,
                                                  electric_potential = e_field,
                                                  animators = animators,
                                                  dipole_gauges = []
                                                  )
        specs.append(spec)

        cp.utils.multi_map(make_movie, specs, processes = 3)
