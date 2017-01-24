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
        # free_state = ion.Superposition(*(ion.FreeSphericalWave(energy = e * eV, l = 1) for e in np.linspace(1, 20, 100)))
        # free_state = ion.HydrogenCoulombState(energy = 1 * eV, l = 0)
        free_state = ion.HydrogenCoulombState(energy = 50 * eV, l = 3)

        r = np.linspace(.01 * bohr_radius, 30 * bohr_radius, 1e4)
        # print(r)

        with cp.utils.Timer() as timer:
            radial_function = free_state.radial_function(r)
        print(timer)
        # print(np.abs(r * radial_function) ** 2)

        print('mem usage: {}'.format(cp.utils.convert_bytes(radial_function.nbytes)))

        cp.utils.xy_plot('r_times_radial_function',
                         r,
                         np.abs(r * radial_function) ** 2,
                         x_scale = 'bohr_radius', x_label = '$r$',
                         y_label = r'$\left|r \, R(r) \right|^2$',
                         target_dir = OUT_DIR)

        cp.utils.xy_plot('radial_function',
                         r,
                         np.abs(radial_function) ** 2,
                         x_scale = 'bohr_radius', x_label = r'$r$',
                         y_label = r'$\left|R(r) \right|^2$',
                         target_dir = OUT_DIR)

        print(free_state)
        print(repr(free_state))

        # sim = ion.SphericalHarmonicSpecification('free_state',
        #                                           r_bound = 200 * bohr_radius, r_points = 200 * 4, l_points = 100,
        #                                           internal_potential = ion.NoPotentialEnergy(),
        #                                           initial_state = free_state,
        #                                           ).to_simulation()
        # print(sim.info())
        # sim.mesh.plot_g(target_dir = OUT_DIR, name_postfix = '_before')

        # sim.run_simulation()
        #
        # sim.mesh.plot_g(target_dir = OUT_DIR, name_postfix = '_after')
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)


        # MEMORY TEST

        free_states = [ion.HydrogenCoulombState(energy = e * eV, l = 0) for e in range(1, 101)]
        mem = []
        for ii, s in enumerate(free_states):
            mem.append(s.radial_function(r))
            print(ii, cp.utils.convert_bytes(sum(x.nbytes for x in mem)))
