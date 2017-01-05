import logging
import os

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization') as logger:
        # free_state = ion.FreeSphericalWave(energy = 5 * eV, l = 0, m = 0)
        free_state = ion.Superposition(*(ion.FreeSphericalWave(energy = e * eV, l = 1) for e in np.linspace(1, 20, 100)))

        print(free_state)
        print(repr(free_state))

        animators = [ion.animators.SphericalHarmonicAnimator(target_dir = OUT_DIR),
                     ion.animators.SphericalHarmonicAnimator(postfix = '30', target_dir = OUT_DIR, plot_limit = 30 * bohr_radius),
                     ion.animators.SphericalHarmonicAnimator(postfix = '100', target_dir = OUT_DIR, plot_limit = 100 * bohr_radius)
                     ]

        spec = ion.SphericalHarmonicSpecification('free_state',
                                                  r_bound = 200 * bohr_radius, r_points = 200 * 4, l_points = 100,
                                                  internal_potential = ion.NoPotentialEnergy(),
                                                  initial_state = free_state,
                                                  # animators = animators
                                                  )

        sim = spec.to_simulation()

        print(sim.info())

        sim.mesh.plot_g(target_dir = OUT_DIR, name_postfix = '_before')

        sim.run_simulation()

        sim.mesh.plot_g(target_dir = OUT_DIR, name_postfix = '_after')
        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
