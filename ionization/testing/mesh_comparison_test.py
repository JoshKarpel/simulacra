import os
import logging

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
BASE_OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logger = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = BASE_OUT_DIR, file_level = logging.DEBUG)


def run(spec):
    with logger as log:
        sim = ion.ElectricFieldSimulation(spec)

        log.info(sim.info())
        sim.run_simulation()
        log.info(sim.info())

        sim.plot_wavefunction_vs_time(save = True, target_dir = spec.extra_args['out_dir'])


if __name__ == '__main__':
    with logger as log:
        n_max = 4

        bound = 50
        points = 2 ** 10
        angular_points = 2 ** 5

        period = 50 * asec

        t_init = -8 * period
        t_final = -t_init

        window = ion.potentials.LinearRampWindow(ramp_on_time = t_init + (200 * asec), ramp_time = 4 * period)
        e_field = ion.potentials.SineWave(omega = twopi / (period), amplitude = 0.5 * atomic_electric_field, window = window)

        specs = []

        for n in range(n_max + 1):
            for l in range(n):
                initial_state = ion.BoundState(n, l, 0)
                OUT_DIR = os.path.join(BASE_OUT_DIR, '{}x{}'.format(points, angular_points), '{}_{}'.format(initial_state.n, initial_state.l))

                ############## CYLINDRICAL SLICE ###################

                cyl_spec = ion.CylindricalSliceSpecification('{}_{}__cyl_slice'.format(n, l),
                                                             time_initial = t_init, time_final = t_final,
                                                             z_points = points, rho_points = points / 2,
                                                             z_bound = bound * bohr_radius, rho_bound = bound * bohr_radius,
                                                             initial_state = initial_state,
                                                             electric_potential = e_field,
                                                             out_dir = OUT_DIR)
                specs.append(cyl_spec)

                ############## SPHERICAL SLICE ###################

                sph_spec = ion.SphericalSliceSpecification('{}_{}__sph_slice'.format(n, l), time_initial = t_init, time_final = t_final,
                                                           r_points = points, theta_points = angular_points,
                                                           r_bound = bound * bohr_radius,
                                                           initial_state = initial_state,
                                                           electric_potential = e_field,
                                                           out_dir = OUT_DIR)
                specs.append(sph_spec)

                ############# SPHERICAL HARMONICS ###################

                sph_harm_spec = ion.SphericalHarmonicSpecification('{}_{}__sph_harm'.format(n, l), time_initial = t_init, time_final = t_final,
                                                                   r_points = points,
                                                                   r_bound = bound * bohr_radius,
                                                                   spherical_harmonics_max_l = angular_points - 1,
                                                                   initial_state = initial_state,
                                                                   electric_potential = e_field,
                                                                   out_dir = OUT_DIR,
                                                                   dipole_gauges = [])
                specs.append(sph_harm_spec)

                ############# LAGRANGIAN SPHERICAL HARMONICS ###################

                lag_sph_harm_spec = ion.SphericalHarmonicSpecification('{}_{}__lag_sph_harm'.format(n, l), time_initial = t_init, time_final = t_final,
                                                                       mesh_type = ion.LagrangianSphericalHarmonicMesh,
                                                                       r_points = points,
                                                                       r_bound = bound * bohr_radius,
                                                                       spherical_harmonics_max_l = angular_points - 1,
                                                                       initial_state = initial_state,
                                                                       electric_potential = e_field,
                                                                       dipole_gauges = [],
                                                                       out_dir = OUT_DIR)
                specs.append(lag_sph_harm_spec)

        cp.utils.multi_map(run, specs, processes = 2)
