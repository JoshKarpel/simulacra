import os

import compy as cp
import compy.quantum.hydrogenic as hyd
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
BASE_OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.INFO, file_logs = True, file_dir = BASE_OUT_DIR, file_level = logging.DEBUG) as logger:
        n_max = 3

        bound = 30
        points = 2 ** 9
        angular_points = 2 ** 7

        t_init = -200 * asec
        t_final = -t_init

        # e_field = hyd.Rectangle(start_time = 20 * asec, end_time = 180 * asec, amplitude = 1 * atomic_electric_field)
        e_field = hyd.SineWave(omega = twopi / (50 * asec), amplitude = 1 * atomic_electric_field, window_time = 100 * asec, window_width = 10 * asec)
        # e_field = None

        for n in range(n_max + 1):
            for l in range(n):
                initial_state = hyd.BoundState(n, l, 0)
                OUT_DIR = os.path.join(BASE_OUT_DIR, '{}_{}'.format(initial_state.n, initial_state.l))

                ############## CYLINDRICAL SLICE ###################

                cyl_spec = hyd.CylindricalSliceSpecification('cyl_slice', time_initial = t_init, time_final = t_final,
                                                             z_points = points, rho_points = points / 2,
                                                             z_bound = bound * bohr_radius, rho_bound = bound * bohr_radius,
                                                             initial_state = initial_state,
                                                             electric_potential = e_field)
                cyl_sim = hyd.ElectricFieldSimulation(cyl_spec)

                logger.info(cyl_sim.info())
                cyl_sim.run_simulation()
                logger.info(cyl_sim.info())

                cyl_sim.plot_wavefunction_vs_time(save = True, target_dir = OUT_DIR)
                # cyl_sim.plot_wavefunction_vs_time(save = True, target_dir = OUT_DIR, grayscale = True)

                ############## SPHERICAL SLICE ###################

                sph_spec = hyd.SphericalSliceSpecification('sph_slice', time_initial = t_init, time_final = t_final,
                                                           r_points = points, theta_points = angular_points,
                                                           r_bound = bound * bohr_radius,
                                                           initial_state = initial_state,
                                                           electric_potential = e_field)
                sph_sim = hyd.ElectricFieldSimulation(sph_spec)

                logger.info(sph_sim.info())
                sph_sim.run_simulation()
                logger.info(sph_sim.info())

                sph_sim.plot_wavefunction_vs_time(save = True, target_dir = OUT_DIR)
                # sph_sim.plot_wavefunction_vs_time(save = True, target_dir = OUT_DIR, grayscale = True)
