import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('compy', 'ionization', stdout_level = logging.DEBUG)


def run_spec(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        return sim


def dense_center(t, spec):
    if 100 * asec < t < 300 * asec:
        return .5 * asec
    else:
        return 1 * asec


if __name__ == '__main__':
    with logman as logger:
        efield = ion.Rectangle(start_time = 100 * asec, end_time = 300 * asec, amplitude = 1 * atomic_electric_field)

        base_spec_kwargs = dict(
                r_bound = 50 * bohr_radius, r_points = 200,
                l_bound = 50,
                time_initial = 0 * asec, time_final = 400 * asec,
                use_numeric_eigenstates_as_basis = True,
                electric_potential = efield,
                electric_potential_dc_correction = True,
        )

        specs = []
        for name, dt in zip(('.5as', '1as', '5as', 'dense_center'),
                            (.5 * asec, 1 * asec, 5 * asec, dense_center)):
            specs.append(ion.SphericalHarmonicSpecification(name,
                                                            time_step = dt,
                                                            **base_spec_kwargs))

        results = si.utils.multi_map(run_spec, specs)

        for r in sorted(results, key = lambda r: r.running_time):
            print(r.running_time, r.name)
