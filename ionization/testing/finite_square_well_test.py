import logging
import os

import numpy as np
import scipy.sparse as sparse

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        mass = electron_mass

        depth = 1.5 * eV
        width = 3 * nm

        z_0 = width * np.sqrt(2 * mass * depth) / hbar
        print('z_0 = {},   floor(z_0 / pi) = {}'.format(z_0, np.floor(z_0 / pi)))

        pot = ion.FiniteSquareWell(potential_depth = depth, width = width)
        init = ion.FiniteSquareWellState(well_depth = depth, well_width = width, mass = mass, n = 1)

        wavenumbers = (twopi / nm) * np.linspace(-10, 10, 1000)
        plane_waves = [ion.OneDFreeParticle(k, mass = mass) for k in wavenumbers]
        dk = np.abs(plane_waves[1].wavenumber - plane_waves[0].wavenumber)

        # electric = ion.SineWave.from_photon_energy(1 * eV, amplitude = .01 * atomic_electric_field,
        #                                            window = ion.SymmetricExponentialTimeWindow(window_time = 10 * fsec, window_width = 1 * fsec, window_center = 5 * fsec))
        electric = ion.NoElectricField()

        ani = [ion.animators.LineAnimator(postfix = '_full_new', target_dir = OUT_DIR, length = 60, renormalize = True)]

        sim = ion.LineSpecification('fsw',
                                    x_bound = 50 * nm, x_points = 2 ** 15,
                                    internal_potential = pot,
                                    electric_potential = electric,
                                    test_mass = mass,
                                    test_states = ion.FiniteSquareWellState.all_states_of_well(depth, width, mass) + plane_waves,
                                    dipole_gauges = (),
                                    initial_state = init,
                                    time_initial = 0 * fsec, time_final = 1 * fsec, time_step = 5 * asec,
                                    mask = ion.RadialCosineMask(inner_radius = 40 * nm, outer_radius = 50 * nm),
                                    # animators = ani
                                    ).to_simulation()

        print(sim.info())

        cp.utils.xy_plot('fsw_potential', sim.mesh.x_mesh, pot(distance = sim.mesh.x_mesh),
                         x_scale = 'nm', y_scale = 'eV',
                         target_dir = OUT_DIR)

        print('init norm', sim.mesh.norm)

        sim.mesh.plot_g(name_postfix = '_init', target_dir = OUT_DIR)

        sim.run_simulation()

        print(sim.info())
        print('norm', sim.mesh.norm)
        print('energy EV', sim.energy_expectation_value_vs_time_internal / eV)

        sim.mesh.plot_g(name_postfix = '_post', target_dir = OUT_DIR)
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, x_scale = 'fsec')

        overlap_vs_k = np.zeros(len(plane_waves)) * np.NaN

        for ii, k in enumerate(sorted(s for s in sim.spec.test_states if s in plane_waves)):
            overlap = sim.state_overlaps_vs_time[k][-1] * dk
            # print('{}: {}'.format(k, overlap))

            overlap_vs_k[ii] = overlap

        print(wavenumbers)
        print(overlap_vs_k)

        print(np.sum(overlap_vs_k))

        cp.utils.xy_plot('overlap_vs_k',
                         wavenumbers, overlap_vs_k,
                         x_scale = twopi / nm, x_label = r'Wavenumber $k$ ($2\pi/\mathrm{nm}$)',
                         y_lower_limit = 0, y_upper_limit = 1,
                         target_dir = OUT_DIR)
