import logging
import itertools as it

import compy as cp
from compy.cluster import *
from compy.units import *
from ionization import core, integrodiff

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ask_mesh_type():
    """
    :return: spec_type, mesh_kwargs
    """
    mesh_kwargs = {}

    mesh_type = cp.cluster.ask_for_input('Mesh Type (cyl | sph | harm)', default = 'harm', cast_to = str)

    try:
        if mesh_type == 'cyl':
            spec_type = core.CylindricalSliceSpecification

            mesh_kwargs['z_bound'] = bohr_radius * cp.cluster.ask_for_input('Z Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['rho_bound'] = bohr_radius * cp.cluster.ask_for_input('Rho Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['z_points'] = 2 * (mesh_kwargs['z_bound'] / bohr_radius) * cp.cluster.ask_for_input('Z Points per Bohr Radii', default = 20, cast_to = int)
            mesh_kwargs['rho_points'] = (mesh_kwargs['rho_bound'] / bohr_radius) * cp.cluster.ask_for_input('Rho Points per Bohr Radii', default = 20, cast_to = int)

            mesh_kwargs['outer_radius'] = max(mesh_kwargs['z_bound'], mesh_kwargs['rho_bound'])

            memory_estimate = (128 / 8) * mesh_kwargs['z_points'] * mesh_kwargs['rho_points']

        elif mesh_type == 'sph':
            spec_type = core.SphericalSliceSpecification

            mesh_kwargs['r_bound'] = bohr_radius * cp.cluster.ask_for_input('R Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['r_points'] = (mesh_kwargs['r_bound'] / bohr_radius) * cp.cluster.ask_for_input('R Points per Bohr Radii', default = 40, cast_to = int)
            mesh_kwargs['theta_points'] = cp.cluster.ask_for_input('Theta Points', default = 100, cast_to = int)

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['theta_points']

        elif mesh_type == 'harm':
            spec_type = core.SphericalHarmonicSpecification

            mesh_kwargs['r_bound'] = bohr_radius * cp.cluster.ask_for_input('R Bound (Bohr radii)', default = 250, cast_to = float)
            mesh_kwargs['r_points'] = (mesh_kwargs['r_bound'] / bohr_radius) * cp.cluster.ask_for_input('R Points per Bohr Radii', default = 8, cast_to = int)
            mesh_kwargs['l_bound'] = cp.cluster.ask_for_input('l points', default = 200, cast_to = int)

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            mesh_kwargs['snapshot_type'] = core.SphericalHarmonicSnapshot

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['l_bound']

        else:
            raise ValueError('Mesh type {} not found!'.format(mesh_type))

        logger.warning('Predicted memory usage per Simulation is >{}'.format(utils.convert_bytes(memory_estimate)))

        return spec_type, mesh_kwargs
    except ValueError:
        ask_mesh_type()


class ElectricFieldSimulationResult(cp.cluster.SimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.time_steps = copy(sim.time_steps)

        state_overlaps = sim.state_overlaps_vs_time

        self.final_norm = copy(sim.norm_vs_time[-1])
        # self.final_state_overlaps = {state: overlap[-1] for state, overlap in state_overlaps.items()}
        self.final_initial_state_overlap = copy(state_overlaps[sim.spec.initial_state][-1])
        self.final_bound_state_overlap = copy(sum(state_overlaps[s] for s in sim.bound_states))
        self.final_free_state_overlap = copy(sum(state_overlaps[s] for s in sim.free_states))

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

        grouped_states, group_labels = sim.group_free_states_by_discrete_attr('l', cutoff_value = 10)
        sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = f'__l__{sim.file_name}',
                                      grouped_free_states = grouped_states, group_labels = group_labels)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l__collapsed',
        #                               collapse_bound_state_angular_momentums = True,
        #                               grouped_free_states = grouped_states, group_labels = group_labels)


class ElectricFieldJobProcessor(cp.cluster.JobProcessor):
    simulation_result_type = ElectricFieldSimulationResult


parameter_name_to_unit_name = {
    'pulse_width': 'asec',
    'fluence': 'Jcm2',
    'phase': 'rad'
}


class ConvergenceSimulationResult(ElectricFieldSimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.r_points = copy(sim.spec.r_points)
        self.delta_r = copy(sim.mesh.delta_r)
        self.delta_t = copy(sim.spec.time_step)


class ConvergenceJobProcessor(ElectricFieldJobProcessor):
    simulation_result_type = ConvergenceSimulationResult


class PulseSimulationResult(ElectricFieldSimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.pulse_type = copy(sim.spec.pulse_type)
        self.pulse_width = copy(sim.spec.pulse_width)
        self.fluence = copy(sim.spec.fluence)
        self.phase = copy(sim.spec.phase)


class PulseJobProcessor(ElectricFieldJobProcessor):
    simulation_result_type = PulseSimulationResult

    def __init__(self, job_name, job_dir_path):
        super().__init__(job_name, job_dir_path, core.ElectricFieldSimulation)

    def make_summary_plots(self):
        super().make_summary_plots()

        if len(self.unprocessed_sim_names) == 0:
            self.make_pulse_parameter_scan_plots()

    def make_pulse_parameter_scan_plots(self):
        for ionization_metric in ('final_norm', 'final_initial_state_overlap', 'final_bound_state_overlap'):
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, line_parameter, scan_parameter in it.permutations(('pulse_width', 'fluence', 'phase')):
                plot_parameter_name, line_parameter_name, scan_parameter_name = plot_parameter.replace('_', ' ').title(), line_parameter.replace('_', ' ').title(), scan_parameter.replace('_', ' ').title()
                plot_parameter_unit, line_parameter_unit, scan_parameter_unit = parameter_name_to_unit_name[plot_parameter], parameter_name_to_unit_name[line_parameter], parameter_name_to_unit_name[scan_parameter]
                plot_parameter_set, line_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(line_parameter), self.parameter_set(scan_parameter)

                for plot_parameter_value in plot_parameter_set:
                    for line_group_number, line_parameter_group in enumerate(cp.utils.grouper(sorted(line_parameter_set), 8)):
                        plot_name = f'{ionization_metric}__{plot_parameter}={uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__grouped_by_{line_parameter}__group_{line_group_number}'

                        lines = []
                        line_labels = []

                        for line_parameter_value in sorted(l for l in line_parameter_group if l is not None):
                            selector = {
                                plot_parameter: plot_parameter_value,
                                line_parameter: line_parameter_value,
                            }
                            results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                            x = np.array([getattr(result, scan_parameter) for result in results])

                            lines.append(np.array([getattr(result, ionization_metric) for result in results]))

                            label = fr"{line_parameter_name}$\, = {uround(line_parameter_value, line_parameter_unit, 3)} \, {unit_names_to_tex_strings[line_parameter_unit]}$"
                            line_labels.append(label)

                        for log in (False, True):
                            if not log:
                                y_lower_limit = 0
                            else:
                                y_lower_limit = None

                            cp.utils.xy_plot(plot_name + f'__log={log}',
                                             x,
                                             *lines,
                                             line_labels = line_labels,
                                             title = f"{plot_parameter_name}$\, = {uround(plot_parameter_value, plot_parameter_unit, 3)} \, {unit_names_to_tex_strings[plot_parameter_unit]}$",
                                             x_label = scan_parameter_name, x_scale = scan_parameter_unit,
                                             y_lower_limit = y_lower_limit, y_upper_limit = 1, y_log_axis = log,
                                             y_label = ionization_metric_name,
                                             legend_on_right = True,
                                             target_dir = self.plots_dir
                                             )


class IDESimulationResult(cp.cluster.SimulationResult):
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


class IDEJobProcessor(cp.cluster.JobProcessor):
    simulation_result_type = IDESimulationResult

    def __init__(self, job_name, job_dir_path):
        super().__init__(job_name, job_dir_path, integrodiff.AdaptiveIntegroDifferentialEquationSimulation)

    def make_summary_plots(self):
        super().make_summary_plots()

        if len(self.unprocessed_sim_names) == 0:
            self.make_pulse_parameter_scan_plots()

    def make_pulse_parameter_scan_plots(self):
        for ionization_metric in ['final_bound_state_overlap']:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, line_parameter, scan_parameter in it.permutations(('pulse_width', 'fluence', 'phase')):
                plot_parameter_name, line_parameter_name, scan_parameter_name = plot_parameter.replace('_', ' ').title(), line_parameter.replace('_', ' ').title(), scan_parameter.replace('_', ' ').title()
                plot_parameter_unit, line_parameter_unit, scan_parameter_unit = parameter_name_to_unit_name[plot_parameter], parameter_name_to_unit_name[line_parameter], parameter_name_to_unit_name[scan_parameter]
                plot_parameter_set, line_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(line_parameter), self.parameter_set(scan_parameter)

                for plot_parameter_value in plot_parameter_set:
                    for line_group_number, line_parameter_group in enumerate(cp.utils.grouper(sorted(line_parameter_set), 8)):
                        plot_name = f'{ionization_metric}__{plot_parameter}={uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__grouped_by_{line_parameter}__group_{line_group_number}'

                        lines = []
                        line_labels = []

                        for line_parameter_value in sorted(l for l in line_parameter_group if l is not None):
                            selector = {
                                plot_parameter: plot_parameter_value,
                                line_parameter: line_parameter_value,
                            }
                            results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                            x = np.array([getattr(result, scan_parameter) for result in results])

                            lines.append(np.array([getattr(result, ionization_metric) for result in results]))

                            label = fr"{line_parameter_name}$\, = {uround(line_parameter_value, line_parameter_unit, 3)} \, {unit_names_to_tex_strings[line_parameter_unit]}$"
                            line_labels.append(label)

                        for log in (False, True):
                            if not log:
                                y_lower_limit = 0
                            else:
                                y_lower_limit = None

                            cp.utils.xy_plot(plot_name + f'__log={log}',
                                             x,
                                             *lines,
                                             line_labels = line_labels,
                                             title = f"{plot_parameter_name}$\, = {uround(plot_parameter_value, plot_parameter_unit, 3)} \, {unit_names_to_tex_strings[plot_parameter_unit]}$",
                                             x_label = scan_parameter_name, x_scale = scan_parameter_unit,
                                             y_lower_limit = y_lower_limit, y_upper_limit = 1, y_log_axis = log,
                                             y_label = ionization_metric_name,
                                             legend_on_right = True,
                                             target_dir = self.plots_dir
                                             )
