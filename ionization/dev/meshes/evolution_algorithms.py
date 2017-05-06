import os
import logging
import itertools as it

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = cp.utils.LogManager('compy', 'ionization', stdout_level = logging.INFO)


def run_spec(spec):
    with logman as logger:
        try:
            sim = spec.to_simulation()

            # logger.info(sim.info())
            sim.run_simulation()
            logger.info(sim.info())

            sim.plot_test_state_overlaps_vs_time(target_dir = OUT_DIR)

            return sim
        except NotImplementedError as e:
            logger.warn(f'{sim.mesh.__class__.__name__} does not support {sim.spec.evolution_method}-{sim.spec.evolution_equations}-{sim.spec.evolution_gauge}')


if __name__ == '__main__':
    with logman as logger:
        efield = ion.SineWave.from_frequency(1 / (100 * asec), amplitude = .1 * atomic_electric_field)

        bound = 25 * bohr_radius

        hyd_spec_base = dict(
                r_bound = bound, rho_bound = bound, z_bound = bound,
                l_bound = 50,
                r_points = 400, theta_points = 400, rho_points = int(bound / bohr_radius) * 10, z_points = int(bound / bohr_radius) * 20,
                initial_state = ion.HydrogenBoundState(2, 0),
                test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
                electric_potential = efield,
                time_initial = 0 * asec, time_final = 500 * asec, time_step = 1 * asec,
        )

        line_potential = ion.HarmonicOscillator.from_energy_spacing_and_mass(1 * eV, electron_mass)
        line_spec_base = hyd_spec_base.copy()
        line_spec_base.update(dict(
                x_bound = bound, x_points = 2 ** 10,
                internal_potential = line_potential,
                electric_potential = ion.SineWave.from_photon_energy(1 * eV, amplitude = .1 * atomic_electric_field),
                initial_state = ion.QHOState.from_potential(line_potential, electron_mass),
                test_states = tuple(ion.QHOState.from_potential(line_potential, electron_mass, n) for n in range(20)),
        ))

        specs = []

        for method, equations, gauge in it.product(
                ('CN', 'SO', 'S'),
                ('HAM', 'LAG'),
                ('LEN', 'VEL')):
            for spec_type in (ion.CylindricalSliceSpecification, ion.SphericalSliceSpecification, ion.SphericalHarmonicSpecification):
                specs.append(
                        spec_type(f'{spec_type.__name__[:-13]}_{method}_{equations}_{gauge}',
                                  **hyd_spec_base,
                                  evolution_method = method, evolution_equations = equations, evolution_gauge = gauge,
                                  )
                )

            specs.append(
                    ion.LineSpecification(f'Line_{method}_{equations}_{gauge}',
                                          **line_spec_base,
                                          evolution_method = method, evolution_equations = equations, evolution_gauge = gauge,
                                          )
            )

        results = cp.utils.multi_map(run_spec, specs)
