import os
import logging

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG, file_logs = False, file_dir = OUT_DIR, file_level = logging.DEBUG)


if __name__ == '__main__':
    with log as logger:

        thetas = np.linspace(0, twopi, 100)
        wavenumbers = np.linspace(.1, 50, 100) * per_nm

        sim = ion.SphericalHarmonicSpecification('snapshots',
                                                 time_initial = 0 * asec, time_final = 100 * asec, time_step = 1 * asec,
                                                 store_data_every = 5,
                                                 r_bound = 50 * bohr_radius, r_points = 50 * 4, l_bound = 50,
                                                 use_numeric_eigenstates_as_basis = True, numeric_eigenstate_energy_max = 50 * eV, numeric_eigenstate_l_max = 10,
                                                 snapshot_indices = [50, -1], snapshot_times = [10 * asec],
                                                 snapshot_type = ion.SphericalHarmonicSnapshot,
                                                 ).to_simulation()

        sim.run_simulation()
        saved_sim_path = sim.save(target_dir = OUT_DIR)

        print(sim.info())

        sim.mesh.plot_electron_momentum_spectrum(target_dir = OUT_DIR, name_postfix = '__from_sim_directly')
        sim.mesh.plot_electron_momentum_spectrum(target_dir = OUT_DIR, name_postfix = '__from_sim_directly_free_only', g_mesh = sim.mesh.get_g_with_states_removed(sim.bound_states))

        loaded_sim = cp.Simulation.load(saved_sim_path)

        for time_index, snapshot in loaded_sim.snapshots.items():
            print(time_index, snapshot, repr(snapshot))

            print(list(snapshot.data.keys()))

            for key, val in snapshot.data.items():
                print(key, val)

            theta, wavenumber, ip = snapshot.data['inner_product_with_plane_waves']
            sim.mesh.plot_electron_momentum_spectrum_from_meshes(theta, wavenumber, ip, 'wavenumber', 'per_nm', target_dir = OUT_DIR, name_postfix = '__snapshot_index={}'.format(snapshot.time_index))

            theta, wavenumber, ip = snapshot.data['inner_product_with_plane_waves__free_only']
            sim.mesh.plot_electron_momentum_spectrum_from_meshes(theta, wavenumber, ip, 'wavenumber', 'per_nm', target_dir = OUT_DIR, name_postfix = '__snapshot_index={}__free_only'.format(snapshot.time_index))

            print()
