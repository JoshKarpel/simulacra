import logging
import os

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:
        n_max = 4

        bound = 50
        points = 2 ** 2

        t_init = -200 * asec
        t_final = -t_init

        # e_field = ion.Rectangle(start_time = 20 * asec, end_time = 180 * asec, amplitude = 1 * atomic_electric_field)
        e_field = ion.potentials.SineWave(omega = twopi / (50 * asec), amplitude = 1 * atomic_electric_field, window_time = 100 * asec, window_width = 10 * asec)
        # e_field = None

        initial_state = ion.BoundState(1, 0, 0)
        test_states = [ion.BoundState(n, l) for n in range(5) for l in range(n)]

        ############## CYLINDRICAL SLICE ###################

        cyl_spec = ion.CylindricalSliceSpecification('cyl_slice',
                                                     time_initial = t_init, time_final = t_final,
                                                     z_points = points, rho_points = points / 2,
                                                     z_bound = bound * bohr_radius, rho_bound = bound * bohr_radius,
                                                     initial_state = initial_state, test_states = test_states,
                                                     electric_potential = e_field)

        cyl_spec.save(target_dir = OUT_DIR)

        cyl_sim = ion.ElectricFieldSimulation(cyl_spec)

        KB = 8e3
        GB = 8e6
        GB = 8e9

        mesh_points = (points ** 2) / 2
        print('Mesh points: {}'.format(mesh_points))

        print('Time Steps: {}'.format(cyl_sim.time_steps))
        print('Test States: {}'.format(len(test_states)))

        mesh_mem = 128 * mesh_points
        time_mem_per_state = 128 * cyl_sim.time_steps
        time_mem = time_mem_per_state * len(test_states)

        print('mesh: {} KB'.format(mesh_mem / KB))
        print('time per state: {} KB'.format(time_mem_per_state / KB))
        print('time total: {} KB'.format(time_mem / KB))
        print('total: {} KB'.format((mesh_mem + time_mem) / KB))

        cyl_sim.file_name = 'cyl_slice__init'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = True)

        cyl_sim.file_name = 'cyl_slice__init_no_mesh'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = False)

        test_sim = cp.Simulation.load(os.path.join(OUT_DIR, 'cyl_slice__init.sim'))
        print(cyl_sim)
        print(test_sim)
        print(test_sim.mesh.get_kinetic_energy_matrix_operators)

        cyl_sim.run_simulation()

        cyl_sim.file_name = 'cyl_slice__done'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = True)

        cyl_sim.file_name = 'cyl_slice__done_no_mesh'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = False)
