import os
import logging

import numpy as np

import compy as cp
import ionization as ion
import plots
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG)


def run(spec):
    with log as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR,
                                             grouped_free_states = {})

        plots.xy_plot(f'{sim.name}__energy_vs_time',
                      sim.times, sim.energy_expectation_value_vs_time_internal,
                      x_label = '$t$', x_scale = 'asec', y_label = 'Energy', y_scale = 'eV',
                      target_dir = OUT_DIR)

if __name__ == '__main__':
    with log as logger:
        energy_spacing = 1 * eV
        mass = electron_mass

        qho = ion.HarmonicOscillator.from_energy_spacing_and_mass(energy_spacing = energy_spacing, mass = mass)

        initial_state = ion.QHOState.from_potential(qho, mass, n = 0)
        test_states = [ion.QHOState.from_potential(qho, mass, n = n) for n in range(31)]

        efield = ion.SineWave.from_photon_energy(energy_spacing, amplitude = .005 * atomic_electric_field)
        # efield = ion.NoElectricField()

        ani_kwargs = dict(
            target_dir = OUT_DIR,
            distance_unit = 'nm',
        )

        ani = [
            ion.animators.LineAnimator(postfix = '_full', **ani_kwargs),
            ion.animators.LineAnimator(postfix = '_zoom', plot_limit = 10 * nm, **ani_kwargs),
        ]

        spec_kwargs = dict(
            x_bound = 50 * nm, x_points = 2 ** 14,
            internal_potential = qho,
            initial_state = initial_state,
            test_states = test_states,
            test_mass = mass,
            electric_potential = efield,
            time_initial = 0,
            time_final = efield.period * 5,
            time_step = 1 * asec,
            animators = ani,
        )

        specs = []
        for method in ('S', 'SO', 'CN'):
            specs.append(ion.LineSpecification(method,
                                               **spec_kwargs,
                                               evolution_method = method,
                                               ))

        cp.utils.multi_map(run, specs, processes = 3)
