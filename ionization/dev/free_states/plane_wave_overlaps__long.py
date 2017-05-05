import logging
import os

from tqdm import tqdm

import numpy as np

import compy as cp
import ionization as ion
import plots
from compy.units import *

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.LogManager('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG, file_logs = False, file_mode = 'w', file_dir = OUT_DIR, file_name = 'log') as logger:
        r_bound = 200
        points_per_bohr_radius = 8

        t_bound = 1000
        t_extra = 1000

        amp = .01
        phase = 0

        window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec)

        # efield = ion.SineWave.from_photon_energy(rydberg + 20 * eV, amplitude = .05 * atomic_electric_field,
        #                                                          window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        efield = ion.SineWave.from_photon_energy(rydberg + 20 * eV, amplitude = amp * atomic_electric_field, phase = phase,
                                                 window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))
        #
        # efield += ion.SineWave.from_photon_energy(rydberg + 30 * eV, amplitude = .05 * atomic_electric_field,
        #                                           window = ion.SymmetricExponentialTimeWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        # efield = ion.SineWave(twopi * (c / (800 * nm)), amplitude = .01 * atomic_electric_field,
        #                       window = window)

        spec_kwargs = dict(
            r_bound = r_bound * bohr_radius,
            r_points = r_bound * points_per_bohr_radius,
            l_bound = 20,
            initial_state = ion.HydrogenBoundState(1, 0),
            time_initial = -t_bound * asec,
            time_final = (t_bound + t_extra) * asec,
            time_step = 1 * asec,
            use_numeric_eigenstates_as_basis = True,
            numeric_eigenstate_energy_max = 10 * eV,
            numeric_eigenstate_l_max = 10,
            electric_potential = efield,
            electric_potential_dc_correction = True,
            mask = ion.RadialCosineMask(inner_radius = .9 * r_bound * bohr_radius, outer_radius = r_bound * bohr_radius),
            store_data_every = 10,
            snapshot_type = ion.SphericalHarmonicSnapshot,
            snapshot_times = [(t_bound + (n * 100)) * asec for n in range(100)],
            snapshot_kwargs = dict(
                plane_wave_overlap__max_wavenumber = 60 * per_nm,
                plane_wave_overlap__wavenumber_points = 200,
                plane_wave_overlap__theta_points = 100,
            ),
        )

        sim = ion.SphericalHarmonicSpecification(f'R={r_bound}_amp={amp}_phase={uround(phase, pi, 3)}pi_tB={t_bound}_tE={t_extra}', **spec_kwargs).to_simulation()

        OUT_DIR = os.path.join(OUT_DIR, sim.name)

        print(sim.info())
        sim.run_simulation()
        sim.save(target_dir = OUT_DIR)
        print(sim.info())

        # sim.mesh.plot_g(target_dir = OUT_DIR)
        # sim.mesh.plot_g(target_dir = OUT_DIR, name_postfix = '_25', plot_limit = 25 * bohr_radius)

        for log in (True, False):
            plots.xy_plot(f'norm_vs_time__log={log}',
                          sim.data_times,
                          sim.norm_vs_time,
                          y_log_axis = log,
                          y_upper_limit = 1,
                          y_label = r'$\left\langle \Psi | \Psi \right\rangle$',
                          x_unit = 'asec',
                          x_label = r'Time $t$',
                          target_dir = OUT_DIR,
                          )

        plot_kwargs = dict(
            target_dir = OUT_DIR,
            bound_state_max_n = 4,
        )

        sim.plot_energy_spectrum(**plot_kwargs,
                                 energy_upper_limit = 50 * eV,
                                 states = 'free',
                                 bins = 25,
                                 group_angular_momentum = False)

        sim.plot_energy_spectrum(**plot_kwargs,
                                 bins = 25,
                                 energy_upper_limit = 50 * eV,
                                 states = 'free',
                                 angular_momentum_cutoff = 10)

        snapshot_spectrum_kwargs = dict(
            target_dir = OUT_DIR,
            # img_format = 'png',
            # img_scale = 3,
        )

        for time_index, snapshot in sim.snapshots.items():
            for log in (True, False):
                theta, wavenumber, ip = snapshot.data['inner_product_with_plane_waves__free_only']

                sim.mesh.plot_electron_momentum_spectrum_from_meshes(theta, wavenumber, ip, 'wavenumber', 'per_nm',
                                                                     log = log,
                                                                     name_postfix = f'_t={uround(snapshot.time, asec, 3)}',
                                                                     **snapshot_spectrum_kwargs)

                plots.xy_plot(f'snapshot_t={uround(snapshot.time, asec, 3)}__free_only__theta=0__log={log}',
                                 wavenumber[0, :], np.abs(ip[0, :]) ** 2,
                              x_unit = 'per_nm', x_label = r'Wavenumber $k$',
                              y_log_axis = log,
                              **snapshot_spectrum_kwargs
                              )
