import matplotlib

matplotlib.use('Agg')

import os
import logging

import compy as cp
from compy.units import *
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def make_movie(spec):
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO,
                         file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w', file_level = logging.INFO) as logger:
        sim = ion.ElectricFieldSimulation(spec)

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)


if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        states = [ion.HydrogenBoundState(n, l) for n in range(3) for l in range(n)]

        bound = 225
        angular_points = 50
        dt = 2.5 * asec

        specs = []

        for amp in (1, 2):
            for period in (100, 200):
                for state in states:
                    prefix = '{}_{}_amp={}_per={}__'.format(state.n, state.l, amp, period)
                    t_init = 0
                    t_final = 20 * period * asec

                    window = ion.LinearRampTimeWindow(ramp_on_time = t_init + period * asec, ramp_time = 5 * period * asec)
                    e_field = ion.SineWave(omega = twopi / (period * asec), amplitude = amp * atomic_electric_field, window = window)
                    internal_potential = ion.Coulomb()
                    mask = ion.RadialCosineMask(inner_radius = (bound - 50) * bohr_radius, outer_radius = bound * bohr_radius)

                    # animators = [ion.animators.CylindricalSliceAnimator(target_dir = OUT_DIR),
                    #              ion.animators.CylindricalSliceAnimator(postfix = 'log', target_dir = OUT_DIR, log = True, renormalize = False),
                    #              ion.animators.CylindricalSliceAnimator(postfix = '30', target_dir = OUT_DIR, plot_limit = 30 * bohr_radius),
                    #              ion.animators.CylindricalSliceAnimator(postfix = '100', target_dir = OUT_DIR, plot_limit = 100 * bohr_radius)]
                    # spec = ion.CylindricalSliceSpecification(prefix + 'cyl_slice', time_initial = t_init, time_final = t_final, time_step = dt,
                    #                                          z_bound = bound * bohr_radius, z_points = bound * 10,
                    #                                          rho_bound = bound * bohr_radius, rho_points = bound * 5,
                    #                                          initial_state = state,
                    #                                          internal_potential = internal_potential,
                    #                                          electric_potential = e_field,
                    #                                          animators = animators
                    #                                          )
                    # specs.append(spec)

                    animators = [ion.animators.SphericalSliceAnimator(target_dir = OUT_DIR),
                                 # ion.animators.SphericalSliceAnimator(postfix = 'log', target_dir = OUT_DIR, log = True, renormalize = False),
                                 ion.animators.SphericalSliceAnimator(postfix = '30', target_dir = OUT_DIR, plot_limit = 30 * bohr_radius),
                                 ion.animators.SphericalSliceAnimator(postfix = '100', target_dir = OUT_DIR, plot_limit = 100 * bohr_radius)
                                 ]
                    # spec = ion.SphericalSliceSpecification(prefix + 'sph_slice_10x180', time_initial = t_init, time_final = t_final, time_step = dt,
                    #                                        r_bound = bound * bohr_radius, r_points = bound * 10,
                    #                                        theta_points = 180,
                    #                                        initial_state = state,
                    #                                        internal_potential = internal_potential,
                    #                                        electric_potential = e_field,
                    #                                        animators = animators
                    #                                        )
                    # specs.append(spec)

                    spec = ion.SphericalSliceSpecification(prefix + 'sph_slice_20x180', time_initial = t_init, time_final = t_final, time_step = dt,
                                                           r_bound = bound * bohr_radius, r_points = bound * 20,
                                                           theta_points = 180,
                                                           initial_state = state,
                                                           internal_potential = internal_potential,
                                                           electric_potential = e_field,
                                                           mask = mask,
                                                           animators = animators
                                                           )
                    specs.append(spec)

                    # spec = ion.SphericalSliceSpecification(prefix + 'sph_slice_10x90', time_initial = t_init, time_final = t_final, time_step = dt,
                    #                                        r_bound = bound * bohr_radius, r_points = bound * 10,
                    #                                        theta_points = 90,
                    #                                        initial_state = state,
                    #                                        internal_potential = internal_potential,
                    #                                        electric_potential = e_field,
                    #                                        animators = animators
                    #                                        )
                    # specs.append(spec)

                    spec = ion.SphericalSliceSpecification(prefix + 'sph_slice_20x90', time_initial = t_init, time_final = t_final, time_step = dt,
                                                           r_bound = bound * bohr_radius, r_points = bound * 20,
                                                           theta_points = 90,
                                                           initial_state = state,
                                                           internal_potential = internal_potential,
                                                           electric_potential = e_field,
                                                           mask = mask,
                                                           animators = animators
                                                           )
                    specs.append(spec)

                    animators = [ion.animators.SphericalHarmonicAnimator(target_dir = OUT_DIR),
                                 # ion.animators.SphericalHarmonicAnimator(postfix = 'log', target_dir = OUT_DIR, log = True, renormalize = False),
                                 ion.animators.SphericalHarmonicAnimator(postfix = '30', target_dir = OUT_DIR, plot_limit = 30 * bohr_radius),
                                 ion.animators.SphericalHarmonicAnimator(postfix = '100', target_dir = OUT_DIR, plot_limit = 100 * bohr_radius)
                                 ]
                    spec = ion.SphericalHarmonicSpecification(prefix + 'harm_CN_4', time_initial = t_init, time_final = t_final, time_step = dt,
                                                              r_bound = bound * bohr_radius, r_points = bound * 4,
                                                              l_points = angular_points,
                                                              initial_state = state,
                                                              internal_potential = internal_potential,
                                                              electric_potential = e_field,
                                                              animators = animators,
                                                              mask = mask,
                                                              evolution_method = 'CN'
                                                              )
                    specs.append(spec)

                    spec = ion.SphericalHarmonicSpecification(prefix + 'harm_SO_4', time_initial = t_init, time_final = t_final, time_step = dt,
                                                              r_bound = bound * bohr_radius, r_points = bound * 4,
                                                              l_points = angular_points,
                                                              initial_state = state,
                                                              internal_potential = internal_potential,
                                                              electric_potential = e_field,
                                                              animators = animators,
                                                              mask = mask,
                                                              evolution_method = 'SO'
                                                              )
                    specs.append(spec)

                    spec = ion.SphericalHarmonicSpecification(prefix + 'harm_CN_10', time_initial = t_init, time_final = t_final, time_step = dt,
                                                              r_bound = bound * bohr_radius, r_points = bound * 10,
                                                              l_points = angular_points,
                                                              initial_state = state,
                                                              internal_potential = internal_potential,
                                                              electric_potential = e_field,
                                                              animators = animators,
                                                              mask = mask,
                                                              evolution_method = 'CN'
                                                              )
                    specs.append(spec)

                    spec = ion.SphericalHarmonicSpecification(prefix + 'harm_SO_10', time_initial = t_init, time_final = t_final, time_step = dt,
                                                              r_bound = bound * bohr_radius, r_points = bound * 10,
                                                              l_points = angular_points,
                                                              initial_state = state,
                                                              internal_potential = internal_potential,
                                                              electric_potential = e_field,
                                                              animators = animators,
                                                              mask = mask,
                                                              evolution_method = 'SO'
                                                              )
                    specs.append(spec)

        cp.utils.multi_map(make_movie, specs, processes = 3)
