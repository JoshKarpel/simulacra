import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME, '_2000nm_1as_19')

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG) as logger:
        mass = electron_mass
        pw = 200

        space_bound = 2000 * nm
        time_bound = 30

        depth = 21.02 * eV
        width = 1.2632 * 2 * bohr_radius

        # depth = 1.5 * eV
        # width = 1 * nm

        z_0 = width * np.sqrt(2 * mass * depth) / hbar / 2
        print('z_0 = {},   floor(z_0 / pi / 2) = {},   pi / 2 = {}'.format(z_0, np.ceil(z_0 / (pi / 2)), pi / 2))

        pot = ion.FiniteSquareWell(potential_depth = depth, width = width)
        init = ion.FiniteSquareWellState(well_depth = depth, well_width = width, mass = mass, n = 1)

        print('energy', init.energy / eV)

        states = list(ion.FiniteSquareWellState.all_states_of_well_from_well(pot, mass))
        for state in states:
            print(state, 'with energy', uround(state.energy, eV), 'eV')

        wavenumbers = (twopi / nm) * np.linspace(-10, 10, 1000)
        plane_waves = [ion.OneDFreeParticle(k, mass = mass) for k in wavenumbers]
        dk = np.abs(plane_waves[1].wavenumber - plane_waves[0].wavenumber)

        # electric = ion.SineWave.from_photon_energy(1 * eV, amplitude = .01 * atomic_electric_field,
        #                                            window = ion.SymmetricExponentialTimeWindow(window_time = 10 * fsec, window_width = 1 * fsec, window_center = 5 * fsec))
        electric = ion.SincPulse(pulse_width = pw * asec, fluence = 1 * Jcm2, phase = 'cos', dc_correction_time = time_bound * pw * asec,
                                 window = ion.SymmetricExponentialTimeWindow(window_time = 29 * pw * asec, window_width = pw * asec / 2))
        # electric = ion.NoElectricField()

        ani = [
            ion.animators.LineAnimator(postfix = '_full', target_dir = OUT_DIR, length = 60, distance_unit = 'bohr_radius', metrics = ('norm', 'initial_state_overlap')),
            ion.animators.LineAnimator(postfix = '_narrow', target_dir = OUT_DIR, length = 60, plot_limit = width * 50, distance_unit = 'bohr_radius', metrics = ('norm', 'initial_state_overlap')),
            ion.animators.LineAnimator(postfix = '_wide', target_dir = OUT_DIR, length = 60, plot_limit = width * 200, distance_unit = 'bohr_radius', metrics = ('norm', 'initial_state_overlap')),
            ion.animators.LineAnimator(postfix = '_narrow__no_renorm', target_dir = OUT_DIR, length = 60, plot_limit = width * 50, distance_unit = 'bohr_radius', renormalize = False, metrics = ('norm', 'initial_state_overlap')),
        ]

        # test_states = ion.FiniteSquareWellState.all_states_of_well_from_parameters(depth, width, mass) + plane_waves
        test_states = ion.FiniteSquareWellState.all_states_of_well_from_parameters(depth, width, mass)

        sim = ion.LineSpecification('fsw',
                                    x_bound = space_bound, x_points = 2 ** 19,
                                    internal_potential = pot,
                                    electric_potential = electric,
                                    test_mass = mass,
                                    test_states = test_states,
                                    dipole_gauges = (),
                                    initial_state = init,
                                    time_initial = -pw * time_bound * asec, time_final = pw * (3 * time_bound) * asec, time_step = 1 * asec,
                                    # mask = ion.RadialCosineMask(inner_radius = width * 100, outer_radius = width * 300),
                                    animators = ani
                                    ).to_simulation()

        print(sim.info())

        cp.utils.xy_plot('fsw_potential', sim.mesh.x_mesh, pot(distance = sim.mesh.x_mesh),
                         x_scale = 'bohr_radius', y_scale = 'eV',
                         target_dir = OUT_DIR)

        print('init norm', sim.mesh.norm)

        sim.mesh.plot_g(name_postfix = '_init', target_dir = OUT_DIR)

        sim.run_simulation()

        print(sim.info())
        print('norm', sim.mesh.norm)
        print('energy EV', sim.energy_expectation_value_vs_time_internal / eV)

        sim.mesh.plot_g(name_postfix = '_post', target_dir = OUT_DIR)
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, x_scale = 'asec')
        cp.utils.xy_plot('energy_vs_time',
                         sim.times, sim.energy_expectation_value_vs_time_internal,
                         x_label = '$t$', x_scale = 'asec', y_label = 'Energy', y_scale = 'eV',
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
