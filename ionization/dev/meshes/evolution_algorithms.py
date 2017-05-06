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

        spec_base = dict(
            r_bound = bound, rho_bound = bound, z_bound = bound,
            l_bound = 50,
            r_points = 400, theta_points = 400, rho_points = int(bound / bohr_radius) * 10, z_points = int(bound / bohr_radius) * 20,
            initial_state = ion.HydrogenBoundState(2, 0),
            test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
            electric_potential = efield,
            time_initial = 0 * asec, time_final = 500 * asec, time_step = 1 * asec,
        )

        specs = []

        for spec_type, method, equations, gauge in it.product(
                (ion.CylindricalSliceSpecification, ion.SphericalSliceSpecification, ion.SphericalHarmonicSpecification),
                ('CN', 'SO', 'S'),
                ('HAM', 'LAG'),
                ('LEN', 'VEL')):
            specs.append(
                spec_type(f'{spec_type.__name__[:-13]}_{method}_{equations}_{gauge}',
                          **spec_base,
                          evolution_method = method, evolution_equations = equations, evolution_gauge = gauge,
                          )
            )

        results = cp.utils.multi_map(run_spec, specs)
