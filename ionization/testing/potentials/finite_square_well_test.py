import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        mass = electron_mass
        pw = 200

        space_bound = 200 * bohr_radius
        time_bound = 30

        depth = 36.831335 * eV
        width = 1 * bohr_radius

        # depth = 5 * eV
        # width = 1 * nm

        z_0 = width * np.sqrt(2 * mass * depth) / hbar / 2
        print('z_0 = {},   ceil(z_0 / pi / 2) = {}'.format(z_0, np.ceil(z_0 / (pi / 2))))

        pot = ion.FiniteSquareWell(potential_depth = depth, width = width)
        # init = ion.FiniteSquareWellState.from_square_well_potential(pot, mass, n = 1) + ion.FiniteSquareWellState.from_square_well_potential(pot, mass = mass, n = 2) + ion.FiniteSquareWellState.from_square_well_potential(pot, mass = mass, n = 3)
        init = ion.FiniteSquareWellState.from_square_well_potential(pot, mass, n = 1)

        # print('energy', init.energy / eV)

        print(rydberg / eV)
        print('States:')
        states = list(ion.FiniteSquareWellState.all_states_of_well_from_well(pot, mass))
        for state in states:
            print(f'{state} with energy {uround(state.energy, eV, 10)} eV, ratio {-state.energy / rydberg}')

        # wavenumbers = (twopi / nm) * np.linspace(-10, 10, 1000)
        # plane_waves = [ion.OneDFreeParticle(k, mass = mass) for k in wavenumbers]
        # dk = np.abs(plane_waves[1].wavenumber - plane_waves[0].wavenumber)

        # electric = ion.SineWave.from_photon_energy(1 * eV, amplitude = .01 * atomic_electric_field,
        #                                            window = ion.SymmetricExponentialTimeWindow(window_time = 10 * fsec, window_width = 1 * fsec, window_center = 5 * fsec))
        electric = ion.SincPulse(pulse_width = pw * asec, fluence = 1 * Jcm2, phase = 0,
                                 window = ion.SymmetricExponentialTimeWindow(window_time = 28 * pw * asec, window_width = .2 * pw * asec))
        # electric = ion.NoElectricField()

        animator_kwargs = {'target_dir': OUT_DIR,
                           'distance_unit': 'bohr_radius',
                           'metrics': ('norm',)}

        ani = [
            ion.animators.LineAnimator(postfix = '_zoom', length = 60, plot_limit = width * 20, **animator_kwargs),
            ion.animators.LineAnimator(postfix = '_zoom_no_renorm', length = 60, plot_limit = width * 20, renormalize = False, **animator_kwargs),
            ion.animators.LineAnimator(postfix = '_full', length = 60, **animator_kwargs),
            ion.animators.LineAnimator(postfix = '_full_no_renorm', length = 60, renormalize = False, **animator_kwargs),
        ]

        # test_states = ion.FiniteSquareWellState.all_states_of_well_from_parameters(depth, width, mass) + plane_waves
        test_states = ion.FiniteSquareWellState.all_states_of_well_from_parameters(depth, width, mass)

        sim = ion.LineSpecification('fsw',
                                    x_bound = space_bound, x_points = 2 ** 14,
                                    internal_potential = pot,
                                    electric_potential = electric,
                                    test_mass = mass,
                                    test_states = test_states,
                                    dipole_gauges = (),
                                    initial_state = init,
                                    # time_initial = 0, time_final = 1000 * asec, time_step = 1 * asec,
                                    time_initial = -pw * time_bound * asec, time_final = pw * (1.5 * time_bound) * asec, time_step = 1 * asec,
                                    mask = ion.RadialCosineMask(inner_radius = space_bound * .9, outer_radius = space_bound),
                                    animators = ani,
                                    evolution_method = 'SO',
                                    ).to_simulation()

        print(sim.info())

        cp.utils.xy_plot('fsw_potential', sim.mesh.x_mesh, pot(distance = sim.mesh.x_mesh),
                         x_scale = 'bohr_radius', y_scale = 'eV', x_lower_limit = -3 * width, x_upper_limit = 3 * width,
                         target_dir = OUT_DIR)

        sim.run_simulation()

        print(sim.info())

        sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR)

        cp.utils.xy_plot('energy_vs_time',
                         sim.times, sim.energy_expectation_value_vs_time_internal,
                         x_label = '$t$', x_scale = 'asec', y_label = 'Energy', y_scale = 'eV',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('energy_vs_time__ratio',
                         sim.times, sim.energy_expectation_value_vs_time_internal / sim.energy_expectation_value_vs_time_internal[0],
                         x_label = '$t$', x_scale = 'asec', y_label = '$E(t) / E(t=0)$',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('energy_vs_time__ratio_log',
                         sim.times, sim.energy_expectation_value_vs_time_internal / sim.energy_expectation_value_vs_time_internal[0],
                         x_label = '$t$', x_scale = 'asec', y_label = '$E(t) / E(t=0)$', y_log_axis = True,
                         target_dir = OUT_DIR)
        # sim.plot_energy_expectation_value_vs_time(target_dir = OUT_DIR, x_scale = 'asec')

        # overlap_vs_k = np.zeros(len(plane_waves)) * np.NaN
        #
        # for ii, k in enumerate(sorted(s for s in sim.spec.test_states if s in plane_waves)):
        #     overlap = sim.state_overlaps_vs_time[k][-1] * dk
        #     # print('{}: {}'.format(k, overlap))
        #
        #     overlap_vs_k[ii] = overlap
        #
        # print(wavenumbers)
        # print(overlap_vs_k)
        #
        # print(np.sum(overlap_vs_k))
        #
        # cp.utils.xy_plot('overlap_vs_k',
        #                  wavenumbers, overlap_vs_k,
        #                  x_scale = twopi / nm, x_label = r'Wavenumber $k$ ($2\pi/\mathrm{nm}$)',
        #                  y_lower_limit = 0, y_upper_limit = 1,
        #                  target_dir = OUT_DIR)
