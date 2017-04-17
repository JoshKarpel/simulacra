import logging
import os

import numpy as np

import compy as cp
import ionization as ion
import plots
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization') as logger:
        # free_state = ion.FreeSphericalWave(energy = 5 * eV, l = 0, m = 0)
        # free_state = ion.Superposition(*(ion.FreeSphericalWave(energy = e * eV, l = 1) for e in np.linspace(1, 20, 100)))
        # free_state = ion.HydrogenCoulombState(energy = 1 * eV, l = 0)
        free_state = ion.HydrogenCoulombState(energy = .000 * rydberg, l = 0)
        # free_state = ion.HydrogenCoulombState.from_wavenumber(k = 10 / nm)

        # r = np.linspace(.01 * bohr_radius, 200 * bohr_radius, 1e4)
        # r = np.linspace(.01 * bohr_radius, 50 * bohr_radius, 1e4) + 5000 * bohr_radius
        r = np.linspace(0, 10 * bohr_radius, 1e3)[1:]

        with cp.utils.BlockTimer() as timer:
            radial_function = free_state.radial_function(r)
            # print(radial_function)
        print('full:', timer)

        # with cp.utils.Timer() as timer:
        #     radial_function_asymptotic = free_state.radial_function_asymptotic(r)
        #     print(radial_function_asymptotic)
        # print('full:', timer)

        print('mem usage: {}'.format(cp.utils.convert_bytes(radial_function.nbytes)))

        print(free_state.epsilon)

        plots.xy_plot('r_times_radial_function_squared',
                      r,
                      np.abs(r * radial_function) ** 2,  # np.abs(r * radial_function_asymptotic) ** 2,
                         # line_labels = ('exact', 'asymp'),
                         x_scale = 'bohr_radius', x_label = '$r$',
                      y_label = r'$\left|r \, R(r) \right|^2$',
                      target_dir = OUT_DIR)

        plots.xy_plot('r_times_radial_function',
                      r,
                      np.real(r * radial_function), np.real(r * radial_function),  # np.abs(r * radial_function_asymptotic) ** 2,
                         line_labels = ('real', 'imag'),
                      x_scale = 'bohr_radius', x_label = '$r$',
                      y_label = r'$r \, R(r)$',
                      target_dir = OUT_DIR)
        #
        plots.xy_plot('radial_function_squared',
                      r,
                      np.abs(radial_function) ** 2,  # np.abs(radial_function_asymptotic) ** 2,
                         # line_labels = ('exact', 'asymp'),
                         x_scale = 'bohr_radius', x_label = r'$r$',
                      y_label = r'$\left|R(r) \right|^2$',
                      target_dir = OUT_DIR)

        plots.xy_plot('radial_function',
                      r,
                      np.real(radial_function), np.imag(radial_function),  # np.abs(radial_function_asymptotic) ** 2,
                         line_labels = ('real', 'imag'),
                      x_scale = 'bohr_radius', x_label = r'$r$',
                      y_label = r'$R(r)$',
                      target_dir = OUT_DIR)

        # print(free_state)
        # print(repr(free_state))
        #
        # for l in range(6):
        #     sim = ion.SphericalHarmonicSpecification('coulomb_state_l={}'.format(l),
        #                                              r_bound = 100 * bohr_radius, r_points = 100 * 4, l_bound = 50,
        #                                              internal_potential = ion.NoPotentialEnergy(),
        #                                              initial_state = ion.HydrogenCoulombState(energy = 50 * eV, l = l),
        #                                              ).to_simulation()
        #     print(repr(sim.spec.initial_state))
        #     sim.mesh.plot_g(target_dir = OUT_DIR)
