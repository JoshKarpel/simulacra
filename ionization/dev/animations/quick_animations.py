import matplotlib

matplotlib.use('Agg')

import os
import logging

import compy as cp
from units import *
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def make_movie(spec):
    with cp.utils.LogManager('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO,
                             file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w', file_level = logging.DEBUG) as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())


if __name__ == '__main__':
    with cp.utils.LogManager('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        specs = []

        bound = 100
        radial_points = bound * 4
        angular_points = 100

        t_init = 0
        t_final = 1000
        dt = 1

        initial_state = ion.HydrogenBoundState(1, 0, 0) + ion.HydrogenBoundState(2, 1, 0)

        window = ion.LinearRampTimeWindow(ramp_on_time = t_init * asec, ramp_time = 200 * asec)
        e_field = ion.SineWave.from_frequency(1 / (50 * asec), amplitude = 1 * atomic_electric_field, window = window)
        mask = ion.RadialCosineMask(inner_radius = (bound - 25) * bohr_radius, outer_radius = bound * bohr_radius)

        animators = [
            ion.animators.CylindricalSliceAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
            ion.animators.CylindricalSliceAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        ]
        specs.append(ion.CylindricalSliceSpecification('cyl_slice',
                                                       time_initial = t_init * asec, time_final = t_final * asec, time_step = dt * asec,
                                                       z_bound = 20 * bohr_radius, z_points = 300,
                                                       rho_bound = 20 * bohr_radius, rho_points = 150,
                                                       initial_state = ion.HydrogenBoundState(1, 0, 0),
                                                       electric_potential = e_field,
                                                       mask = ion.RadialCosineMask(inner_radius = 15 * bohr_radius, outer_radius = 20 * bohr_radius),
                                                       animators = animators
                                                       ))

        animators = [
            ion.animators.SphericalSliceAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
            ion.animators.SphericalSliceAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        ]
        specs.append(ion.SphericalSliceSpecification('sph_slice',
                                                     time_initial = t_init * asec, time_final = t_final * asec, time_step = dt * asec,
                                                     r_bound = bound * bohr_radius, r_points = radial_points,
                                                     theta_points = angular_points,
                                                     initial_state = initial_state,
                                                     electric_potential = e_field,
                                                     mask = mask,
                                                     animators = animators
                                                     ))

        animators = [
            ion.animators.SphericalHarmonicAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
            ion.animators.SphericalHarmonicAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        ]
        specs.append(ion.SphericalHarmonicSpecification('sph_harm', time_initial = t_init * asec, time_final = t_final * asec, time_step = dt * asec,
                                                        r_bound = bound * bohr_radius, r_points = radial_points,
                                                        l_bound = angular_points,
                                                        initial_state = initial_state,
                                                        electric_potential = e_field,
                                                        mask = mask,
                                                        animators = animators,
                                                        ))

        #######

        mass = electron_mass
        pot = ion.HarmonicOscillator.from_energy_spacing_and_mass(energy_spacing = 1 * eV, mass = mass)

        init = ion.QHOState.from_potential(pot, mass, n = 1) + ion.QHOState.from_potential(pot, mass, n = 2)

        animators = [
            ion.animators.LineAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
            ion.animators.LineAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        ]
        specs.append(ion.LineSpecification('line',
                                           x_bound = 50 * nm, x_points = 2 ** 14,
                                           internal_potential = pot,
                                           electric_potential = ion.SineWave.from_photon_energy(1 * eV, amplitude = .05 * atomic_electric_field),
                                           test_states = (ion.QHOState.from_potential(pot, mass, n = n) for n in range(20)),
                                           initial_state = init,
                                           time_initial = t_init * asec, time_final = t_final * 10 * asec, time_step = dt * asec,
                                           mask = ion.RadialCosineMask(inner_radius = 40 * nm, outer_radius = 50 * nm),
                                           animators = animators
                                           ))

        cp.utils.multi_map(make_movie, specs, processes = 4)
