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
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG,
                         file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w', file_level = logging.DEBUG) as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        bound = 200
        radial_points = bound * 4
        angular_points = 100

        specs = []

        t_init = 0
        t_final = 50
        dt = 1

        initial_state = ion.HydrogenBoundState(2, 1, 0)

        window = ion.LinearRampTimeWindow(ramp_on_time = t_init * asec, ramp_time = (t_init + 200) * asec)
        e_field = ion.SineWave.from_frequency(1 / (100 * asec), amplitude = 1 * atomic_electric_field, window = window)
        # e_field = ion.SineWave.from_photon_energy(1 * eV, amplitude = .1 * atomic_electric_field, window = window)
        mask = ion.RadialCosineMask(inner_radius = (bound - 50) * bohr_radius, outer_radius = bound * bohr_radius)

        animators = [
            ion.animators.CylindricalSliceAnimator(target_dir = OUT_DIR),
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
            ion.animators.SphericalSliceAnimator(target_dir = OUT_DIR),
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
            ion.animators.SphericalHarmonicAnimator(target_dir = OUT_DIR),
        ]
        specs.append(ion.SphericalHarmonicSpecification('sph_harm', time_initial = t_init * asec, time_final = t_final * asec, time_step = dt * asec,
                                                        r_bound = bound * bohr_radius, r_points = radial_points,
                                                        l_points = angular_points,
                                                        initial_state = initial_state,
                                                        electric_potential = e_field,
                                                        mask = mask,
                                                        animators = animators,
                                                        ))

        cp.utils.multi_map(make_movie, specs, processes = 3)
