import itertools as it
import logging
from copy import copy

import numpy as np
import numpy.ma as ma

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

from ionization import core, integrodiff


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ask_mesh_type():
    """
    :return: spec_type, mesh_kwargs
    """
    mesh_kwargs = {}

    mesh_type = clu.ask_for_input('Mesh Type (cyl | sph | harm)', default = 'harm', cast_to = str)

    try:
        if mesh_type == 'cyl':
            spec_type = core.CylindricalSliceSpecification

            mesh_kwargs['z_bound'] = bohr_radius * clu.ask_for_input('Z Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['rho_bound'] = bohr_radius * clu.ask_for_input('Rho Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['z_points'] = 2 * (mesh_kwargs['z_bound'] / bohr_radius) * clu.ask_for_input('Z Points per Bohr Radii', default = 20, cast_to = int)
            mesh_kwargs['rho_points'] = (mesh_kwargs['rho_bound'] / bohr_radius) * clu.ask_for_input('Rho Points per Bohr Radii', default = 20, cast_to = int)

            mesh_kwargs['outer_radius'] = max(mesh_kwargs['z_bound'], mesh_kwargs['rho_bound'])

            memory_estimate = (128 / 8) * mesh_kwargs['z_points'] * mesh_kwargs['rho_points']

        elif mesh_type == 'sph':
            spec_type = core.SphericalSliceSpecification

            mesh_kwargs['r_bound'] = bohr_radius * clu.ask_for_input('R Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['r_points'] = (mesh_kwargs['r_bound'] / bohr_radius) * clu.ask_for_input('R Points per Bohr Radii', default = 40, cast_to = int)
            mesh_kwargs['theta_points'] = clu.ask_for_input('Theta Points', default = 100, cast_to = int)

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['theta_points']

        elif mesh_type == 'harm':
            spec_type = core.SphericalHarmonicSpecification

            mesh_kwargs['r_bound'] = bohr_radius * clu.ask_for_input('R Bound (Bohr radii)', default = 250, cast_to = float)
            mesh_kwargs['r_points'] = (mesh_kwargs['r_bound'] / bohr_radius) * clu.ask_for_input('R Points per Bohr Radii', default = 8, cast_to = int)
            mesh_kwargs['l_bound'] = clu.ask_for_input('l points', default = 200, cast_to = int)

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            mesh_kwargs['snapshot_type'] = core.SphericalHarmonicSnapshot

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['l_bound']

        else:
            raise ValueError('Mesh type {} not found!'.format(mesh_type))

        logger.warning('Predicted memory usage per Simulation is >{}'.format(si.utils.bytes_to_str(memory_estimate)))

        return spec_type, mesh_kwargs
    except ValueError:
        ask_mesh_type()


parameter_name_to_unit_name = {
    'pulse_width': 'asec',
    'fluence': 'Jcm2',
    'phase': 'rad',
    'delta_r': 'bohr_radius',
    'delta_t': 'asec',
    }


class PulseParameterScanMixin:
    scan_parameters = ['pulse_width', 'fluence', 'phase']

    def make_summary_plots(self):
        super().make_summary_plots()

        if len(self.unprocessed_sim_names) == 0:
            logger.info(f'Generating pulse parameter scans for job {self.name}')
            # self.make_pulse_parameter_scans_1d()
            self.make_pulse_parameter_scans_2d()

    def make_pulse_parameter_scans_1d(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, line_parameter, scan_parameter in it.permutations(self.scan_parameters):
                plot_parameter_name, line_parameter_name, scan_parameter_name = plot_parameter.replace('_', ' ').title(), line_parameter.replace('_', ' ').title(), scan_parameter.replace('_', ' ').title()
                plot_parameter_unit, line_parameter_unit, scan_parameter_unit = parameter_name_to_unit_name[plot_parameter], parameter_name_to_unit_name[line_parameter], parameter_name_to_unit_name[scan_parameter]
                plot_parameter_set, line_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(line_parameter), self.parameter_set(scan_parameter)

                if len(scan_parameter_set) < 2:
                    continue

                for plot_parameter_value in plot_parameter_set:
                    for line_group_number, line_parameter_group in enumerate(si.utils.grouper(sorted(line_parameter_set), 8)):
                        plot_name = f'{ionization_metric}__{plot_parameter}={uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__grouped_by_{line_parameter}__group_{line_group_number}'

                        lines = []
                        line_labels = []

                        for line_parameter_value in sorted(l for l in line_parameter_group if l is not None):
                            selector = {
                                plot_parameter: plot_parameter_value,
                                line_parameter: line_parameter_value,
                                }
                            results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                            lines.append(np.array([getattr(result, ionization_metric) for result in results]))

                            label = fr"{line_parameter_name}$\, = {uround(line_parameter_value, line_parameter_unit, 3)} \, {UNIT_NAME_TO_LATEX[line_parameter_unit]}$"
                            line_labels.append(label)

                        x = np.array([getattr(result, scan_parameter) for result in results])

                        for log_x, log_y in it.product((True, False), repeat = 2):
                            if scan_parameter == 'phase':
                                log_x = False

                            if not log_y:
                                y_upper_limit = 1
                                y_lower_limit = 0
                            else:
                                y_upper_limit = None
                                y_lower_limit = None

                            if any((log_x, log_y)):
                                log_str = '__log'

                                if log_x:
                                    log_str += 'X'

                                if log_y:
                                    log_str += 'Y'
                            else:
                                log_str = ''

                            si.plots.xy_plot('1d__' + plot_name + log_str,
                                             x,
                                             *lines,
                                             line_labels = line_labels,
                                             title = f"{plot_parameter_name}$\, = {uround(plot_parameter_value, plot_parameter_unit, 3)} \, {UNIT_NAME_TO_LATEX[plot_parameter_unit]}$",
                                             x_label = scan_parameter_name, x_unit = scan_parameter_unit,
                                             y_lower_limit = y_lower_limit, y_upper_limit = y_upper_limit, y_log_axis = log_y, x_log_axis = log_x,
                                             y_label = ionization_metric_name,
                                             legend_on_right = True,
                                             target_dir = self.summaries_dir

                                             )

    def make_pulse_parameter_scans_2d(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            # for plot_parameter, x_parameter, y_parameter in it.permutations(self.scan_parameters):
            for plot_parameter in self.scan_parameters:
                for x_parameter, y_parameter in it.combinations((p for p in self.scan_parameters if p != plot_parameter), r = 2):  # overkill, but whatever
                    plot_parameter_name, x_parameter_name, y_parameter_name = plot_parameter.replace('_', ' ').title(), x_parameter.replace('_', ' ').title(), y_parameter.replace('_', ' ').title()
                    plot_parameter_unit, x_parameter_unit, y_parameter_unit = parameter_name_to_unit_name[plot_parameter], parameter_name_to_unit_name[x_parameter], parameter_name_to_unit_name[y_parameter]
                    plot_parameter_set, x_parameter_set, y_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(x_parameter), self.parameter_set(y_parameter)

                    if len(x_parameter_set) < 2 or len(y_parameter_set) < 2:
                        continue

                    x, y = np.array(sorted(x_parameter_set)), np.array(sorted(y_parameter_set))
                    x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

                    for plot_parameter_value in plot_parameter_set:
                        plot_name = f'{ionization_metric}__{plot_parameter}={uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__{x_parameter}_vs_{y_parameter}'

                        results = self.select_by_kwargs(**{plot_parameter: plot_parameter_value})

                        xy_to_metric = {(getattr(r, x_parameter), getattr(r, y_parameter)): getattr(r, ionization_metric) for r in results}
                        z_mesh = np.zeros(x_mesh.shape) * np.NaN

                        for ii, x_value in enumerate(x):
                            for jj, y_value in enumerate(y):
                                z_mesh[ii, jj] = xy_to_metric[(x_value, y_value)]

                        for log_x, log_y in it.product((True, False), repeat = 2):
                            # skip log phase plots
                            if (x_parameter == 'phase' and log_x) or (y_parameter == 'phase' and log_y):
                                continue

                            if any((log_x, log_y)):
                                log_str = '__log'

                                if log_x:
                                    log_str += 'X'

                                if log_y:
                                    log_str += 'Y'
                            else:
                                log_str = ''

                            z_lower_limit = np.nanmin(z_mesh)
                            z_upper_limit = 1

                            plot_name = '2d__' + plot_name + log_str

                            try:
                                si.plots.xyz_plot(plot_name,
                                                  x_mesh, y_mesh, z_mesh,
                                                  x_unit = x_parameter_unit, y_unit = y_parameter_unit,
                                                  x_label = x_parameter_name, y_label = y_parameter_name,
                                                  x_log_axis = log_x, y_log_axis = log_y,
                                                  z_log_axis = True, z_lower_limit = z_lower_limit, z_upper_limit = z_upper_limit,
                                                  z_label = f"{ionization_metric_name} for {plot_parameter_name}$\, = {uround(plot_parameter_value, plot_parameter_unit, 3)} \, {UNIT_NAME_TO_LATEX[plot_parameter_unit]}$",
                                                  target_dir = self.summaries_dir)
                            except ValueError as e:
                                logger.warning(f'Failed to make plot {plot_name} because of {e}')


class ElectricFieldSimulationResult(clu.SimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.time_steps = copy(sim.time_steps)

        state_overlaps = sim.state_overlaps_vs_time

        self.final_norm = copy(sim.norm_vs_time[-1])
        self.final_initial_state_overlap = copy(state_overlaps[sim.spec.initial_state][-1])
        self.final_bound_state_overlap = copy(sum(state_overlaps[s][-1] for s in sim.bound_states))
        self.final_free_state_overlap = copy(sum(state_overlaps[s][-1] for s in sim.free_states))

        if len(sim.data_times) > 2:
            self.make_wavefunction_plots(sim)

    def make_wavefunction_plots(self, sim):
        plot_kwargs = dict(
                target_dir = self.plots_dir,
                plot_name = 'name',
                show_title = True,
                )

        # sim.plot_wavefunction_vs_time(**plot_kwargs)

        grouped_states, group_labels = sim.group_free_states_by_continuous_attr('energy', divisions = 12, cutoff_value = 100 * eV, label_unit = 'eV')
        sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = f'__energy__{sim.file_name}',
                                      grouped_free_states = grouped_states, group_labels = group_labels)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__energy__collapsed',
        #                               collapse_bound_state_angular_momentums = True,
        #                               grouped_free_states = grouped_states, group_labels = group_labels)

        try:
            grouped_states, group_labels = sim.group_free_states_by_discrete_attr('l', cutoff_value = 10)
            sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = f'__l__{sim.file_name}',
                                          grouped_free_states = grouped_states, group_labels = group_labels)
            # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l__collapsed',
            #                               collapse_bound_state_angular_momentums = True,
            #                               grouped_free_states = grouped_states, group_labels = group_labels)
        except AttributeError:  # free states must not have l
            pass


class ElectricFieldJobProcessor(PulseParameterScanMixin, clu.JobProcessor):
    simulation_result_type = ElectricFieldSimulationResult

    ionization_metrics = ['final_norm', 'final_initial_state_overlap', 'final_bound_state_overlap']

    def __init__(self, job_name, job_dir_path):
        super().__init__(job_name, job_dir_path, core.ElectricFieldSimulation)


class ConvergenceSimulationResult(ElectricFieldSimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.r_points = copy(sim.spec.r_points)
        self.r_bound = copy(sim.spec.r_bound)
        self.delta_r = self.r_bound / self.r_points
        self.delta_t = copy(sim.spec.time_step)


class ConvergenceJobProcessor(ElectricFieldJobProcessor):
    simulation_result_type = ConvergenceSimulationResult

    scan_parameters = ['delta_r', 'delta_t']

    def make_summary_plots(self):
        super().make_summary_plots()

        if len(self.unprocessed_sim_names) == 0:
            logger.info(f'Generating relative pulse parameter scans for job {self.name}')
            self.make_pulse_parameter_scans_1d_relative()
            self.make_pulse_parameter_scans_2d_relative()

    def make_pulse_parameter_scans_1d(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, scan_parameter in it.permutations(self.scan_parameters):
                plot_parameter_name, scan_parameter_name = plot_parameter.replace('_', ' ').title(), scan_parameter.replace('_', ' ').title()
                plot_parameter_unit, scan_parameter_unit = parameter_name_to_unit_name[plot_parameter], parameter_name_to_unit_name[scan_parameter]
                plot_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(scan_parameter)

                for plot_parameter_value in plot_parameter_set:
                    plot_name = f'{ionization_metric}__{plot_parameter}={uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}'

                    selector = {
                        plot_parameter: plot_parameter_value,
                        }
                    results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                    line = np.array([getattr(result, ionization_metric) for result in results])

                    x = np.array([getattr(result, scan_parameter) for result in results])

                    for log_x, log_y in it.product((False, True), repeat = 2):
                        if any((log_x, log_y)):
                            log_str = '__log'

                            if log_x:
                                log_str += 'X'

                            if log_y:
                                log_str += 'Y'
                        else:
                            log_str = ''

                        if not log_y:
                            y_upper_limit = 1
                            y_lower_limit = 0
                        else:
                            y_upper_limit = None
                            y_lower_limit = None

                        si.plots.xy_plot('1d__' + plot_name + log_str,
                                         x,
                                         line,
                                         title = f"{plot_parameter_name}$\, = {uround(plot_parameter_value, plot_parameter_unit, 3)} \, {UNIT_NAME_TO_LATEX[plot_parameter_unit]}$",
                                         x_label = scan_parameter_name, x_unit = scan_parameter_unit, x_log_axis = log_x,
                                         y_lower_limit = y_lower_limit, y_upper_limit = y_upper_limit, y_log_axis = log_y,
                                         y_label = ionization_metric_name,
                                         legend_on_right = True,
                                         target_dir = self.summaries_dir
                                         )

    def make_pulse_parameter_scans_1d_relative(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, scan_parameter in it.permutations(self.scan_parameters):
                plot_parameter_name, scan_parameter_name = plot_parameter.replace('_', ' ').title(), scan_parameter.replace('_', ' ').title()
                plot_parameter_unit, scan_parameter_unit = parameter_name_to_unit_name[plot_parameter], parameter_name_to_unit_name[scan_parameter]
                plot_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(scan_parameter)

                for plot_parameter_value in plot_parameter_set:
                    plot_name = f'{ionization_metric}__{plot_parameter}={uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__rel'

                    selector = {
                        plot_parameter: plot_parameter_value,
                        }
                    results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                    line = np.array([np.abs(getattr(result, ionization_metric) - getattr(results[0], ionization_metric)) for result in results])
                    line = ma.masked_less_equal(line, 0)

                    x = np.array([getattr(result, scan_parameter) for result in results])

                    for log_x, log_y in it.product((False, True), repeat = 2):
                        if any((log_x, log_y)):
                            log_str = '__log'

                            if log_x:
                                log_str += 'X'

                            if log_y:
                                log_str += 'Y'
                        else:
                            log_str = ''

                        si.plots.xy_plot('1d__' + plot_name + log_str,
                                         x,
                                         line,
                                         title = f"{plot_parameter_name}$\, = {uround(plot_parameter_value, plot_parameter_unit, 3)} \, {UNIT_NAME_TO_LATEX[plot_parameter_unit]}$ (Diff from Best)",
                                         x_label = scan_parameter_name, x_unit = scan_parameter_unit, x_log_axis = log_x,
                                         y_lower_limit = None, y_upper_limit = None, y_log_axis = log_y,
                                         y_label = ionization_metric_name,
                                         legend_on_right = True,
                                         target_dir = self.summaries_dir
                                         )

    def make_pulse_parameter_scans_2d(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for x_parameter, y_parameter in it.combinations(self.scan_parameters, r = 2):
                x_parameter_name, y_parameter_name = x_parameter.replace('_', ' ').title(), y_parameter.replace('_', ' ').title()
                x_parameter_unit, y_parameter_unit = parameter_name_to_unit_name[x_parameter], parameter_name_to_unit_name[y_parameter]
                x_parameter_set, y_parameter_set = self.parameter_set(x_parameter), self.parameter_set(y_parameter)

                plot_name = ionization_metric

                x = np.array(sorted(x_parameter_set))
                y = np.array(sorted(y_parameter_set))

                x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

                xy_to_metric = {(getattr(r, x_parameter), getattr(r, y_parameter)): getattr(r, ionization_metric) for r in self.data.values()}
                z_mesh = np.zeros(x_mesh.shape) * np.NaN

                for ii, x_value in enumerate(x):
                    for jj, y_value in enumerate(y):
                        z_mesh[ii, jj] = xy_to_metric[(x_value, y_value)]

                for log_x, log_y, log_z in it.product((True, False), repeat = 3):
                    if any((log_x, log_y, log_z)):
                        log_str = '__log'

                        if log_x:
                            log_str += 'X'

                        if log_y:
                            log_str += 'Y'

                        if log_z:
                            log_str += 'Z'
                    else:
                        log_str = ''

                    if log_z:
                        z_lower_limit = np.nanmin(z_mesh)
                        z_upper_limit = np.nanmax(z_mesh)
                    else:
                        z_lower_limit = 0
                        z_upper_limit = 1

                    si.plots.xyz_plot('2d__' + plot_name + log_str,
                                      x_mesh, y_mesh, z_mesh,
                                      x_unit = x_parameter_unit, y_unit = y_parameter_unit,
                                      x_label = x_parameter_name, y_label = y_parameter_name,
                                      x_log_axis = log_x, y_log_axis = log_y, z_log_axis = log_z,
                                      z_lower_limit = z_lower_limit, z_upper_limit = z_upper_limit,
                                      z_label = ionization_metric_name,
                                      target_dir = self.summaries_dir)

    def make_pulse_parameter_scans_2d_relative(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for x_parameter, y_parameter in it.combinations(self.scan_parameters, r = 2):
                x_parameter_name, y_parameter_name = x_parameter.replace('_', ' ').title(), y_parameter.replace('_', ' ').title()
                x_parameter_unit, y_parameter_unit = parameter_name_to_unit_name[x_parameter], parameter_name_to_unit_name[y_parameter]
                x_parameter_set, y_parameter_set = self.parameter_set(x_parameter), self.parameter_set(y_parameter)

                plot_name = f'{ionization_metric}__{x_parameter}_x_{y_parameter}__rel'

                x = np.array(sorted(x_parameter_set))
                y = np.array(sorted(y_parameter_set))

                x_min = np.nanmin(x)
                y_min = np.nanmin(y)

                x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

                xy_to_metric = {(getattr(r, x_parameter), getattr(r, y_parameter)): getattr(r, ionization_metric) for r in self.data.values()}
                z_mesh = np.zeros(x_mesh.shape) * np.NaN

                best = xy_to_metric[(x_min, y_min)]

                for ii, x_value in enumerate(x):
                    for jj, y_value in enumerate(y):
                        z_mesh[ii, jj] = np.abs(xy_to_metric[(x_value, y_value)] - best)

                z_mesh = ma.masked_less_equal(z_mesh, 0)

                for log_x, log_y, log_z in it.product((True, False), repeat = 3):
                    if any((log_x, log_y, log_z)):
                        log_str = '__log'

                        if log_x:
                            log_str += 'X'

                        if log_y:
                            log_str += 'Y'

                        if log_z:
                            log_str += 'Z'
                    else:
                        log_str = ''

                    si.plots.xyz_plot('2d__' + plot_name + log_str,
                                      x_mesh, y_mesh, z_mesh,
                                      x_unit = x_parameter_unit, y_unit = y_parameter_unit,
                                      x_label = x_parameter_name, y_label = y_parameter_name,
                                      x_log_axis = log_x, y_log_axis = log_y, z_log_axis = log_z,
                                      z_lower_limit = None, z_upper_limit = None,
                                      z_label = ionization_metric_name + ' (Diff from Best)',
                                      target_dir = self.summaries_dir)


class PulseSimulationResult(ElectricFieldSimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.pulse_type = copy(sim.spec.pulse_type)
        self.pulse_width = copy(sim.spec.pulse_width)
        self.fluence = copy(sim.spec.fluence)
        self.phase = copy(sim.spec.phase)

        self.pulse_window = copy(sim.spec.electric_potential.window.window_time)


class PulseJobProcessor(ElectricFieldJobProcessor):
    simulation_result_type = PulseSimulationResult

    ionization_metrics = ['final_norm', 'final_initial_state_overlap', 'final_bound_state_overlap']


class IDESimulationResult(clu.SimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.pulse_type = copy(sim.spec.pulse_type)
        self.pulse_width = copy(sim.spec.pulse_width)
        self.fluence = copy(sim.spec.fluence)
        self.phase = copy(sim.spec.phase)

        self.final_bound_state_overlap = np.abs(sim.a[-1]) ** 2

        self.make_a_plots(sim)

    def make_a_plots(self, sim):
        plot_kwargs = dict(
                target_dir = self.plots_dir,
                plot_name = 'name',
                show_title = True,
                name_postfix = f'__{sim.file_name}',
                )

        sim.plot_a_vs_time(**plot_kwargs)
        sim.plot_a_vs_time(**plot_kwargs, log = True)


class IDEJobProcessor(PulseParameterScanMixin, clu.JobProcessor):
    simulation_result_type = IDESimulationResult

    ionization_metrics = ['final_bound_state_overlap']

    def __init__(self, job_name, job_dir_path):
        super().__init__(job_name, job_dir_path, integrodiff.AdaptiveIntegroDifferentialEquationSimulation)
