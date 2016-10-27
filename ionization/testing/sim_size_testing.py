import logging
import os

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger(stdout_level = logging.INFO, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG) as logger:
        bound = 50
        points = 2 ** 13
        angular_points = 128

        t_init = 0
        t_final = 10000 * asec

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

        KB = 8e3 * 1.024
        MB = 8e6 * 1.024
        GB = 8e9 * 1.024

        mesh_points = (points ** 2) / 2
        print('Mesh points: {}'.format(mesh_points))

        print('Time Steps: {}'.format(cyl_sim.time_steps))
        print('Test States: {}'.format(len(test_states)))

        mesh_mem = 128 * mesh_points
        pos_mem = 128 * 2 * mesh_points
        coord_mem = 128 * 2 * points
        memo_operators = 128 * points * 3 * 5
        memo_meshes = 128 * mesh_points * len(test_states)
        time_mem_per_state = 128 * cyl_sim.time_steps
        time_mem = time_mem_per_state * len(test_states)
        extra_mem = 128 * cyl_sim.time_steps * 3  # norm, electric field, energy expectation

        print('mesh: {} KB'.format(mesh_mem / KB))
        print('pos: {} KB'.format(pos_mem / KB))
        print('coord: {} KB'.format(coord_mem / KB))
        print('memo operators: {} KB'.format(memo_operators / KB))
        print('memo meshes: {} KB'.format(memo_meshes / KB))
        print('time per state: {} KB'.format(time_mem_per_state / KB))
        print('time total: {} KB'.format(time_mem / KB))
        print('extra: {} KB'.format(extra_mem / KB))
        print('total init: {} KB'.format((mesh_mem + time_mem + extra_mem) / KB))

        done_mem = time_mem + mesh_mem + extra_mem
        done_mem += mesh_mem * (len(test_states))

        print('total done: {} KB'.format(done_mem / KB))

        cyl_sim.file_name = 'cyl_slice__init'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = True)

        cyl_sim.file_name = 'cyl_slice__init_no_mesh'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = False)

        with cp.utils.Timer() as t:
            cyl_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        with cp.utils.Timer() as t:
            cyl_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        test_sim = cp.Simulation.load(os.path.join(OUT_DIR, 'cyl_slice__init.sim'))
        # print(cyl_sim)
        # print(test_sim)
        # print(test_sim.mesh.get_kinetic_energy_matrix_operators)
        # for k, v in sorted(vars(test_sim).items()):
        #     print(k, '  :  ', v)

        with cp.utils.Timer() as t:
            test_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        with cp.utils.Timer() as t:
            test_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        for k, v in sorted(vars(test_sim.mesh).items()):
            print(k, '  :  ', v)

        # with cp.utils.Timer() as t:
        #     cyl_sim.run_simulation()
        # print('sim run time', t)

        cyl_sim.file_name = 'cyl_slice__done'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = True)

        cyl_sim.file_name = 'cyl_slice__done_no_mesh'
        cyl_sim.save(target_dir = OUT_DIR, save_mesh = False)

        ############## SPHERICAL SLICE ###################

        sph_spec = ion.SphericalSliceSpecification('sph_slice',
                                                   time_initial = t_init, time_final = t_final,
                                                   r_points = points, theta_points = angular_points,
                                                   r_bound = bound * bohr_radius,
                                                   initial_state = initial_state, test_states = test_states,
                                                   electric_potential = e_field)

        sph_spec.save(target_dir = OUT_DIR)

        sph_sim = ion.ElectricFieldSimulation(sph_spec)

        KB = 8e3 * 1.024
        MB = 8e6 * 1.024
        GB = 8e9 * 1.024

        mesh_points = (points ** 2) / 2
        print('Mesh points: {}'.format(mesh_points))

        print('Time Steps: {}'.format(sph_sim.time_steps))
        print('Test States: {}'.format(len(test_states)))

        mesh_mem = 128 * mesh_points
        pos_mem = 128 * 2 * mesh_points
        coord_mem = 128 * 2 * points
        memo_operators = 128 * points * 3 * 5
        memo_meshes = 128 * mesh_points * len(test_states)
        time_mem_per_state = 128 * sph_sim.time_steps
        time_mem = time_mem_per_state * len(test_states)
        extra_mem = 128 * sph_sim.time_steps * 3  # norm, electric field, energy expectation

        print('mesh: {} KB'.format(mesh_mem / KB))
        print('pos: {} KB'.format(pos_mem / KB))
        print('coord: {} KB'.format(coord_mem / KB))
        print('memo operators: {} KB'.format(memo_operators / KB))
        print('memo meshes: {} KB'.format(memo_meshes / KB))
        print('time per state: {} KB'.format(time_mem_per_state / KB))
        print('time total: {} KB'.format(time_mem / KB))
        print('extra: {} KB'.format(extra_mem / KB))
        print('total init: {} KB'.format((mesh_mem + time_mem + extra_mem) / KB))

        done_mem = time_mem + mesh_mem + extra_mem
        done_mem += mesh_mem * (len(test_states))

        print('total done: {} KB'.format(done_mem / KB))

        sph_sim.file_name = 'sph_slice__init'
        sph_sim.save(target_dir = OUT_DIR, save_mesh = True)

        sph_sim.file_name = 'sph_slice__init_no_mesh'
        sph_sim.save(target_dir = OUT_DIR, save_mesh = False)

        with cp.utils.Timer() as t:
            sph_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        with cp.utils.Timer() as t:
            sph_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        test_sim = cp.Simulation.load(os.path.join(OUT_DIR, 'sph_slice__init.sim'))
        # print(sph_sim)
        # print(test_sim)
        # print(test_sim.mesh.get_kinetic_energy_matrix_operators)
        # for k, v in sorted(vars(test_sim).items()):
        #     print(k, '  :  ', v)

        with cp.utils.Timer() as t:
            test_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        with cp.utils.Timer() as t:
            test_sim.mesh.get_kinetic_energy_matrix_operators()

        print(t)

        for k, v in sorted(vars(test_sim.mesh).items()):
            print(k, '  :  ', v)

        # with cp.utils.Timer() as t:
        #     sph_sim.run_simulation()
        # print('sim run time', t)

        sph_sim.file_name = 'sph_slice__done'
        sph_sim.save(target_dir = OUT_DIR, save_mesh = True)

        sph_sim.file_name = 'sph_slice__done_no_mesh'
        sph_sim.save(target_dir = OUT_DIR, save_mesh = False)
