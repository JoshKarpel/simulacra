import logging
import os
import itertools as it

import numpy as np

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LIB_DIR = os.path.join(OUT_DIR, 'sim_lib')

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG)


def run_spec(spec):
    with log as logger:
        try:
            sim = ion.ElectricFieldSimulation.load(os.path.join(LIB_DIR, spec.name + '.sim'))
        except FileNotFoundError:
            sim = spec.to_simulation()

            logger.info(sim.info())
            sim.run_simulation()
            sim.save(target_dir = LIB_DIR)
            logger.info(sim.info())

        # if 'reference' in sim.name:
        #     sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        return sim


def plot_final_norm_vs_r_points(name, sims, ref, **kwargs):
    sims = list(sims)  # consume iterators

    r_points = [sim.spec.r_points_per_br for sim in sims][:-1]
    final_norms = [sim.norm_vs_time[-1] - ref.norm_vs_time[-1] for sim in sims][:-1]

    base_name = name + '__final_norm_vs_r_points_per_br'

    for log in (False, True):
        plot_name = base_name
        if log:
            plot_name += '__log'

        cp.utils.xy_plot(plot_name,
                         r_points,
                         final_norms,
                         x_label = 'Radial Points per Bohr Radius',
                         y_label = 'Relative Final Wavefunction Norm',
                         y_log_axis = log,
                         **kwargs)


def plot_final_initial_state_vs_r_points(name, sims, ref, **kwargs):
    sims = list(sims)  # consume iterators

    r_points = [sim.spec.r_points_per_br for sim in sims][:-1]
    final_initial_overlaps = [np.abs(sim.state_overlaps_vs_time[sim.spec.initial_state][-1] - ref.state_overlaps_vs_time[sim.spec.initial_state][-1]) for sim in sims][:-1]

    base_name = name + '__final_initial_state_vs_r_points_per_br'

    for log in (False, True):
        plot_name = base_name
        if log:
            plot_name += '__log'

        cp.utils.xy_plot(plot_name,
                         r_points,
                         final_initial_overlaps,
                         x_label = 'Radial Points per Bohr Radius',
                         y_label = 'Relative Final Initial State Overlap',
                         y_log_axis = log,
                         **kwargs)


def plot_final_norm_vs_time_step(name, sims, ref, **kwargs):
    sims = list(sims)  # consume iterators

    time_steps = [sim.spec.time_step for sim in sims][:-1]
    final_norms = [sim.norm_vs_time[-1] - ref.norm_vs_time[-1] for sim in sims][:-1]

    base_name = name + '__final_norm_vs_time_step'

    for log_x in (False, True):
        for log_y in (False, True):
            plot_name = base_name
            if log_x:
                plot_name += '__logX'
            if log_y:
                plot_name += '__logY'

            cp.utils.xy_plot(plot_name,
                             time_steps,
                             final_norms,
                             x_label = 'Time Step', x_scale = 'asec',
                             y_label = 'Relative Final Wavefunction Norm',
                             y_log_axis = log_y, x_log_axis = log_x,
                             **kwargs)


def plot_final_initial_state_vs_time_step(name, sims, ref, **kwargs):
    sims = list(sims)  # consume iterators

    time_steps = [sim.spec.time_step for sim in sims][:-1]
    final_norms = [np.abs(sim.state_overlaps_vs_time[sim.spec.initial_state][-1] - ref.state_overlaps_vs_time[sim.spec.initial_state][-1]) for sim in sims][:-1]

    base_name = name + '__final_initial_state_vs_time_step'

    for log_x in (False, True):
        for log_y in (False, True):
            plot_name = base_name
            if log_x:
                plot_name += '__logX'
            if log_y:
                plot_name += '__logY'

            cp.utils.xy_plot(plot_name,
                             time_steps,
                             final_norms,
                             x_label = 'Time Step', x_scale = 'asec',
                             y_label = 'Relative Final Initial State Overlap',
                             y_log_axis = log_y, x_log_axis = log_x,
                             **kwargs)


if __name__ == '__main__':
    with log as logger:
        r_bound = 100
        t_bound = 150  # symmetric around 0
        amp = .1

        electric_potential = ion.Rectangle(start_time = -100 * asec, end_time = 100 * asec, amplitude = amp * atomic_electric_field)

        l_points = 50

        # r_points_per_br_list = [1, 2, 4]
        # time_step_list = [5, 2, 1]

        # r_points_per_br_list = [1, 2, 4, 5, 8, 10, 16, 20, 32]
        # time_step_list = [10, 7.5, 5, 2.5, 2, 1, .75, .5, .25, .2, .1, .075, .05, .025, .02, .01]

        r_points_per_br_list = range(1, 32 + 1)
        t = np.array([10, 7.5, 5, 2.5, 2])
        time_step_list = np.concatenate([t, t / 10, t / 100, t / 1000])

        prefix = 'R={}br__L={}__amp={}aef'.format(r_bound, l_points, amp)

        specs = []

        # specs.append(ion.SphericalHarmonicSpecification('reference',
        #                                                 r_bound = r_bound * bohr_radius,
        #                                                 r_points = r_bound * r_points_per_br_list[-1],
        #                                                 l_points = l_points,
        #                                                 time_step = time_step_list[-1] * asec,
        #                                                 time_initial = -t_bound * asec,
        #                                                 time_final = t_bound * asec,
        #                                                 electric_potential = electric_potential,
        #                                                 use_numeric_eigenstates_as_basis = True,
        #                                                 numeric_eigenstate_energy_max = 10 * eV,
        #                                                 numeric_eigenstate_l_max = 0,
        #                                                 store_data_every = 1,
        #                                                 r_points_per_br = r_points_per_br_list[-1],
        #                                                 dt = time_step_list[-1],
        #                                                 ))

        for r_points_per_br, time_step in it.product(r_points_per_br_list, time_step_list):
            spec_name = [prefix] + ['{}x{}'.format(r_points_per_br, l_points), 'dt={}as'.format(time_step)]

            specs.append(ion.SphericalHarmonicSpecification('__'.join(spec_name),
                                                            r_bound = r_bound * bohr_radius,
                                                            r_points = r_bound * r_points_per_br,
                                                            l_points = l_points,
                                                            time_step = time_step * asec,
                                                            time_initial = -t_bound * asec,
                                                            time_final = t_bound * asec,
                                                            electric_potential = electric_potential,
                                                            use_numeric_eigenstates_as_basis = True,
                                                            numeric_eigenstate_energy_max = 10 * eV,
                                                            numeric_eigenstate_l_max = 0,
                                                            store_data_every = 0,
                                                            r_points_per_br = r_points_per_br,
                                                            dt = time_step,
                                                            ))

        sims = cp.utils.multi_map(run_spec, specs, processes = 6)

        # with open(os.path.join('ref_info.txt', mode = 'w') as f:
        #     print(sims[0].info(), file = f)

        r_points_per_br_set = set(sim.spec.r_points_per_br for sim in sims)
        time_steps_set = set(sim.spec.dt for sim in sims)

        sims_dict = {(sim.spec.r_points_per_br, sim.spec.dt): sim for sim in sims}

        ### CONVERGENCE PLOTS ###

        OUT_DIR_PLT = os.path.join(OUT_DIR, prefix)

        ref = sims_dict[max(r_points_per_br_set), min(time_steps_set)]

        for r_points_per_br in sorted(r_points_per_br_set):
            plot_final_norm_vs_time_step('r_points_per_br={}__smallest_dt={}as'.format(r_points_per_br, min(time_steps_set)), [sims_dict[r_points_per_br, time_step] for time_step in sorted(time_steps_set)], ref, target_dir = OUT_DIR_PLT)
            plot_final_initial_state_vs_time_step('r_points_per_br={}__smallest_dt={}as'.format(r_points_per_br, min(time_steps_set)), [sims_dict[r_points_per_br, time_step] for time_step in sorted(time_steps_set)], ref, target_dir = OUT_DIR_PLT)

        for time_step in sorted(time_steps_set):
            plot_final_norm_vs_r_points('dt={}as__max_ppbr={}'.format(time_step, max(r_points_per_br_set)), [sims_dict[r_points_per_br, time_step] for r_points_per_br in sorted(r_points_per_br_set)], ref, target_dir = OUT_DIR_PLT)
            plot_final_initial_state_vs_r_points('dt={}as__max_ppbr={}'.format(time_step, max(r_points_per_br_set)), [sims_dict[r_points_per_br, time_step] for r_points_per_br in sorted(r_points_per_br_set)], ref, target_dir = OUT_DIR_PLT)
