import os
import logging

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG)

if __name__ == '__main__':
    with log as logger:
        energy_spacing = 1 * eV
        mass = electron_mass

        qho = ion.HarmonicOscillator.from_energy_spacing_and_mass(energy_spacing = energy_spacing, mass = mass)

        initial_state = ion.QHOState.from_potential(qho, mass, n = 0)
        test_states = [ion.QHOState.from_potential(qho, mass, n = n) for n in range(31)]

        # efield = ion.SineWave.from_photon_energy(energy_spacing, amplitude = .005 * atomic_electric_field)
        efield = ion.NoElectricField()

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
            # time_final = efield.period * 5,
            time_final = 100 * asec,
            time_step = 1 * asec,
            animators = ani,
            analytic_eigenstate_type = ion.QHOState,
            use_numeric_eigenstates_as_basis = True,
            numeric_eigenstate_energy_max = 100 * eV,
        )

        sim = ion.LineSpecification('eig',
                                    evolution_method = 'CN',
                                    **spec_kwargs).to_simulation()

        for numeric_eigenstate in sorted(sim.spec.test_states, key = lambda s: s.energy):
            # print(numeric_eigenstate, numeric_eigenstate.analytic_state)

            name = f'{uround(numeric_eigenstate.energy, "eV", 3)}eV'

            cp.utils.xy_plot(name,
                             sim.mesh.x_mesh,
                             np.abs(numeric_eigenstate(sim.mesh.x_mesh)) ** 2, np.abs(numeric_eigenstate.analytic_state(sim.mesh.x_mesh)) ** 2,
                             line_labels = ('numeric ', 'analytic'),
                             x_label = r'$x$', x_scale = 'nm', x_lower_limit = -10 * nm, x_upper_limit = 10 * nm,
                             target_dir = OUT_DIR, img_format = 'png', img_scale = 3)
