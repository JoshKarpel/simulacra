import logging
import os
import itertools as it

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.DEBUG)


def run_spec(spec):
    with log as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        return sim


PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
)


def plot_final_norm_vs_r_points(name, sims):
    sims = list(sims)  # consume iterators

    r_points = [sim.spec.r_points_per_br for sim in sims]
    final_norms = [sim.norm_vs_time[-1] for sim in sims]

    base_name = name + '__final_norm_vs_r_points_per_br'

    for log in (False, True):
        if log:
            plot_name = base_name + '__log'
        else:
            plot_name = base_name

        cp.utils.xy_plot(plot_name,
                         r_points,
                         final_norms,
                         x_label = 'Radial Points per Bohr Radius',
                         y_label = 'Final Wavefunction Norm',
                         y_log_axis = log,
                         **PLOT_KWARGS)


def plot_final_norm_vs_time_step(name, sims):
    sims = list(sims)  # consume iterators

    time_steps = [sim.spec.time_step for sim in sims]
    final_norms = [sim.norm_vs_time[-1] for sim in sims]

    base_name = name + '__final_norm_vs_time_step'

    for log_x in (False, True):
        for log_y in (False, True):
            if log_x:
                plot_name = base_name + '__logX'
            if log_y:
                plot_name = base_name + '__logY'
            else:
                plot_name = base_name

            cp.utils.xy_plot(plot_name,
                             time_steps,
                             final_norms,
                             x_label = 'Time Step', x_scale = 'asec',
                             y_label = 'Final Wavefunction Norm',
                             y_log_axis = log_y, x_log_axis = log_x,
                             **PLOT_KWARGS)


if __name__ == '__main__':
    with log as logger:
        r_bound = 100
        t_bound = 50  # symmetric around 0

        electric_potential = ion.Rectangle(start_time = -t_bound / 2 * asec, end_time = t_bound / 2 * asec, amplitude = 1 * atomic_electric_field)

        l_points = 50
        r_points_per_br_list = range(1, 20)
        time_step_list = [10, 7.5, 5, 2.5, 2, 1, .5, .2, .1, .05, .02, .01]

        base_name = ['R={}br'.format(r_bound)]

        specs = []
        for r_points_per_br, time_step in it.product(r_points_per_br_list, time_step_list):
            spec_name = base_name + ['{}x{}'.format(r_points_per_br, l_points), 'dt={}as'.format(time_step)]

            specs.append(ion.SphericalHarmonicSpecification('__'.join(spec_name),
                                                            r_bound = r_bound * bohr_radius,
                                                            r_points = r_bound * r_points_per_br,
                                                            l_points = l_points,
                                                            time_step = time_step * asec,
                                                            time_initial = -t_bound * asec,
                                                            time_final = t_bound * asec,
                                                            # use_numeric_eigenstates_as_basis = True,
                                                            # numeric_eigenstate_energy_max = 50 * eV,
                                                            # numeric_eigenstate_l_max = 5,
                                                            store_data_every = 0,
                                                            r_points_per_br = r_points_per_br,
                                                            dt = time_step,
                                                            ))

        sims = cp.utils.multi_map(run_spec, specs, processes = 4)

        r_points_per_br_set = set(sim.spec.r_points_per_br for sim in sims)
        time_steps_set = set(sim.spec.dt for sim in sims)

        sims_dict = {(sim.spec.r_points_per_br, sim.spec.dt): sim for sim in sims}

        ### CONVERGENCE PLOTS ###

        for r_points_per_br in sorted(r_points_per_br_set):
            plot_final_norm_vs_time_step('r_points_per_br={}'.format(r_points_per_br), [sims_dict[r_points_per_br, time_step] for time_step in sorted(time_steps_set)])

        for time_step in sorted(time_steps_set):
            plot_final_norm_vs_r_points('dt={}as'.format(time_step), [sims_dict[r_points_per_br, time_step] for r_points_per_br in sorted(r_points_per_br_set)])
