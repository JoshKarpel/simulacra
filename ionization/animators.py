import functools
import logging
import os
import itertools as it
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import compy as cp
from compy.units import *
from . import core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MetricsAndElectricField(cp.AxisManager):
    def __init__(self, axis, simulation,
                 log_metrics = False, time_unit = 'asec', electric_field_unit = 'AEF', metrics = ('norm',),
                 label_top = False, label_left = True, ticks_top = False, legend_kwargs = None):
        self.time_unit_str = ''
        if type(time_unit) == str:
            self.time_unit_str = UNIT_NAMES_TO_TEX[time_unit]
            time_unit = UNIT_NAMES_TO_VALUES[time_unit]
        self.time_unit = time_unit

        if type(electric_field_unit) == str:
            self.electric_field_unit_str = UNIT_NAMES_TO_TEX[electric_field_unit]
            self.electric_field_unit = UNIT_NAMES_TO_VALUES[electric_field_unit]
        else:
            self.electric_field_unit_str = ''
            self.electric_field_unit = electric_field_unit

        self.log_metrics = log_metrics

        self.label_top = label_top
        self.label_left = label_left
        self.ticks_top = ticks_top
        self.legend_kwargs = legend_kwargs

        self.metrics = metrics

        super(MetricsAndElectricField, self).__init__(axis, simulation)

    def initialize(self):
        self.time_line, = self.axis.plot([self.sim.data_times[self.sim.data_time_index] / self.time_unit,
                                          self.sim.data_times[self.sim.data_time_index] / self.time_unit],
                                         [0, 2],
                                         color = 'gray',
                                         animated = True)

        self.redraw += [self.time_line]

        self._initialize_electric_field()

        for metric in self.metrics:
            self.__getattribute__('_initialize_metric_' + metric)()

        legend_options = {'loc': 'lower left',
                          'fontsize': 20,
                          'fancybox': True,
                          'framealpha': .1}
        if self.legend_kwargs is not None:
            legend_options.update(self.legend_kwargs)

        self.legend = self.axis.legend(**legend_options)
        self.redraw += [self.legend]

        self.axis.grid(True, color = 'gray', linestyle = '--')

        self.axis.set_xlabel(r'Time $t$ (${}$)'.format(self.time_unit_str), fontsize = 24)

        if self.label_left:
            self.axis.set_ylabel('Wavefunction Metric', fontsize = 24)

        if self.label_top:
            self.axis.xaxis.set_label_position('top')

        self.axis.tick_params(labeltop = self.ticks_top)

        self.axis.set_xlim(self.sim.data_times[0] / self.time_unit, self.sim.data_times[-1] / self.time_unit)
        if self.log_metrics:
            self.axis.set_yscale('log')
            self.axis.set_ylim(1e-8, 1)
        else:
            self.axis.set_ylim(0, 1.025)
        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]

        super(MetricsAndElectricField, self).initialize()

    def _initialize_electric_field(self):
        self.axis_field = self.axis.twinx()

        y_limit = 1.05 * np.nanmax(np.abs(self.spec.electric_potential.get_electric_field_amplitude(self.sim.data_times))) / self.electric_field_unit
        self.axis_field.set_ylim(-y_limit, y_limit)

        self.axis_field.set_ylabel(r'${}(t)$ (${}$)'.format(str_efield, self.electric_field_unit_str),
                                   fontsize = 24, color = '#d62728')
        self.axis_field.yaxis.set_label_position('right')
        self.axis_field.tick_params(axis = 'both', which = 'major', labelsize = 14)
        self.axis_field.grid(True, color = '#d62728', linestyle = ':')

        for tick in self.axis_field.get_yticklabels():
            tick.set_color('#d62728')

        self.electric_field_line, = self.axis_field.plot(self.sim.data_times / self.time_unit,
                                                         self.sim.electric_field_amplitude_vs_time / self.electric_field_unit,
                                                         label = r'$E(t)$ ({})'.format(self.electric_field_unit_str),
                                                         color = core.COLOR_ELECTRIC_FIELD, linewidth = 3,
                                                         animated = True)

        self.redraw += [self.electric_field_line, *self.axis_field.xaxis.get_gridlines(), *self.axis_field.yaxis.get_gridlines()]

    def _initialize_metric_norm(self):
        self.norm_line, = self.axis.plot(self.sim.data_times / self.time_unit,
                                         self.sim.norm_vs_time,
                                         label = r'$\left\langle \psi|\psi \right\rangle$',
                                         color = 'black', linewidth = 3,
                                         animated = True)

        self.redraw += [self.norm_line]

    def _initialize_metric_initial_state_overlap(self):
        self.initial_state_overlap_line, = self.axis.plot(self.sim.data_times / self.time_unit,
                                                          self.sim.state_overlaps_vs_time[self.sim.spec.initial_state],
                                                          label = r'$\left| \left\langle \psi|{} \right\rangle \right|^2$'.format(self.sim.spec.initial_state.tex_str),
                                                          color = 'blue', linewidth = '3',
                                                          animated = True)

        self.redraw += [self.initial_state_overlap_line]

    def update(self):
        self._update_electric_field()

        for metric in self.metrics:
            self.__getattribute__('_update_metric_' + metric)()

        self.time_line.set_xdata([self.sim.data_times[self.sim.data_time_index] / self.time_unit, self.sim.data_times[self.sim.data_time_index] / self.time_unit])

        super(MetricsAndElectricField, self).update()

    def _update_electric_field(self):
        self.electric_field_line.set_ydata(self.sim.electric_field_amplitude_vs_time / self.electric_field_unit)

    def _update_metric_norm(self):
        self.norm_line.set_ydata(self.sim.norm_vs_time)

    def _update_metric_initial_state_overlap(self):
        self.initial_state_overlap_line.set_ydata(self.sim.state_overlaps_vs_time[self.sim.spec.initial_state])


class TestStateStackplot(cp.AxisManager):
    def __init__(self, axis, simulation,
                 log_metrics = False, time_unit = 'asec',
                 label_top = False, label_left = True, ticks_top = True, ticks_right = True, legend_kwargs = None):
        self.time_unit_str = ''
        if type(time_unit) == str:
            self.time_unit_str = UNIT_NAMES_TO_TEX[time_unit]
            time_unit = UNIT_NAMES_TO_VALUES[time_unit]
        self.time_unit = time_unit

        self.log_metrics = log_metrics

        self.label_top = label_top
        self.label_left = label_left
        self.ticks_top = ticks_top
        self.ticks_right = ticks_right
        self.legend_kwargs = legend_kwargs

        super().__init__(axis, simulation)

    def initialize(self):
        self._initialize_stackplot()

        self.time_line, = self.axis.plot([self.sim.data_times[self.sim.data_time_index] / self.time_unit,
                                          self.sim.data_times[self.sim.data_time_index] / self.time_unit],
                                         [0, 2],
                                         color = 'gray',
                                         animated = True)

        self.redraw += [self.time_line]

        legend_options = {'bbox_to_anchor': (1., -.2),
                          'loc': 'upper right',
                          'borderaxespad': 0.,
                          'fontsize': 20,
                          'fancybox': True,
                          'framealpha': .1}
        if self.legend_kwargs is not None:
            legend_options.update(self.legend_kwargs)

        self.legend = self.axis.legend(**legend_options)
        self.redraw += [self.legend]

        self.axis.grid(True, color = 'gray', linestyle = '--')

        self.axis.set_xlabel(r'Time $t$ ({})'.format(self.time_unit_str), fontsize = 24)

        # if self.label_left:
        #     self.axis.set_ylabel('Wavefunction Metric', fontsize = 24)

        if self.label_top:
            self.axis.xaxis.set_label_position('top')

        self.axis.tick_params(labeltop = self.ticks_top, labelright = self.ticks_right)

        self.axis.set_xlim(self.sim.data_times[0] / self.time_unit, self.sim.data_times[-1] / self.time_unit)
        if self.log_metrics:
            self.axis.set_yscale('log')
            self.axis.set_ylim(1e-8, 1)
        else:
            self.axis.set_ylim(0, 1.025)
        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]

        super().initialize()

    def _get_stackplot_data(self):
        return [self.sim.state_overlaps_vs_time[state] for state in self.spec.test_states]

    def _initialize_stackplot(self):
        self.overlaps_stackplot = self.axis.stackplot(self.sim.data_times / self.time_unit,
                                                      *self._get_stackplot_data(),
                                                      labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2$'.format(state.tex_str) for state in self.spec.test_states],
                                                      animated = True)

        self.redraw += [*self.overlaps_stackplot]

    def _update_stackplot_lines(self):
        for x in self.overlaps_stackplot:
            self.redraw.remove(x)
            x.remove()

        self.axis.set_color_cycle(None)
        self.overlaps_stackplot = self.axis.stackplot(self.sim.data_times / self.time_unit,
                                                      *self._get_stackplot_data(),
                                                      labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2$'.format(state.tex_str) for state in self.spec.test_states],
                                                      animated = True)

        self.redraw = [*self.overlaps_stackplot] + self.redraw

    def update(self):
        self._update_stackplot_lines()

        self.time_line.set_xdata([self.sim.data_times[self.sim.data_time_index] / self.time_unit, self.sim.data_times[self.sim.data_time_index] / self.time_unit])

        super().update()


class QuantumMeshAxis(cp.AxisManager):
    def __init__(self, axis, simulation,
                 plot_limit = None,
                 renormalize = True,
                 log_g = False,
                 overlay_probability_current = False,
                 distance_unit = 'bohr_radius'):
        self.plot_limit = plot_limit
        self.renormalize = renormalize
        self.log_g = log_g
        self.overlay_probability_current = overlay_probability_current
        self.distance_unit = distance_unit

        super(QuantumMeshAxis, self).__init__(axis, simulation)


class WavefunctionSimulationAnimator(cp.Animator):
    def __init__(self, *args,
                 plot_limit = None,
                 renormalize = True,
                 log_g = False,
                 log_metrics = False,
                 overlay_probability_current = False,
                 distance_unit = 'bohr_radius',
                 metrics = ('norm',),
                 **kwargs):
        super(WavefunctionSimulationAnimator, self).__init__(*args, **kwargs)

        self.plot_limit = plot_limit
        self.renormalize = renormalize
        self.log_g = log_g
        self.log_metrics = log_metrics
        self.overlay_probability_current = overlay_probability_current
        self.distance_unit = distance_unit
        self.metrics = metrics

    def __str__(self):
        return cp.utils.field_str(self, 'postfix', ('plot_limit', self.distance_unit), 'distance_unit', 'renormalize', 'log_g', 'log_metrics', 'overlay_probability_current')

    def __repr__(self):
        return self.__str__()


class LineAxis(QuantumMeshAxis):
    def initialize(self):
        unit_value, unit_name = get_unit_value_and_tex_from_unit(self.distance_unit)

        self.mesh = self.sim.mesh.attach_g_to_axis(self.axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, distance_unit = self.distance_unit, animated = True)
        self.redraw += [self.mesh]

        self.axis.grid(True, color = cp.plots.COLOR_OPPOSITE_INFERNO, linestyle = ':')  # change grid color to make it show up against the colormesh

        self.axis.set_xlabel(r'$x$ (${}$)'.format(unit_name), fontsize = 24)
        self.axis.set_ylabel(r'$\left|\psi\right|^2$', fontsize = 30)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)
        self.axis.tick_params(labelright = True, labeltop = True)

        self.axis.axis('tight')

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        super(LineAxis, self).initialize()

    def update(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)

        super(LineAxis, self).update()


class LineAnimator(WavefunctionSimulationAnimator):
    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))

        self.ax_mesh = LineAxis(self.fig.add_axes([.07, .34, .88, .62]), self.sim,
                                plot_limit = self.plot_limit,
                                renormalize = self.renormalize,
                                log_g = self.log_g,
                                overlay_probability_current = self.overlay_probability_current,
                                distance_unit = self.distance_unit)
        self.ax_metrics = MetricsAndElectricField(self.fig.add_axes([.065, .065, .85, .2]), self.sim,
                                                  log_metrics = self.log_metrics,
                                                  metrics = self.metrics)

        self.axis_managers += [self.ax_mesh, self.ax_metrics]

        super(LineAnimator, self)._initialize_figure()


class CylindricalSliceAxis(QuantumMeshAxis):
    def initialize(self):
        unit_value, unit_name = get_unit_value_and_tex_from_unit(self.distance_unit)

        self.mesh = self.sim.mesh.attach_g_to_axis(self.axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, distance_unit = self.distance_unit, animated = True)
        self.redraw += [self.mesh]

        if self.overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(self.axis, plot_limit = self.plot_limit, distance_unit = self.distance_unit, animated = True)
            self.redraw += [self.quiver]

        self.axis.grid(True, color = cp.plots.COLOR_OPPOSITE_INFERNO, linestyle = ':', linewidth = 2)  # change grid color to make it show up against the colormesh

        self.axis.set_xlabel(r'$z$ ({})'.format(unit_name), fontsize = 24)
        self.axis.set_ylabel(r'$\rho$ ({})'.format(unit_name), fontsize = 24)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)

        divider = make_axes_locatable(self.axis)
        cax = divider.append_axes("right", size = "2%", pad = 0.05)
        self.cbar = plt.colorbar(cax = cax, mappable = self.mesh)
        self.cbar.ax.tick_params(labelsize = 20)

        self.axis.axis('tight')

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines(), *self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        super(CylindricalSliceAxis, self).initialize()

    def update(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)

        super(CylindricalSliceAxis, self).update()


class CylindricalSliceAnimator(WavefunctionSimulationAnimator):
    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))

        self.ax_mesh = CylindricalSliceAxis(self.fig.add_axes([.07, .34, .88, .62]), self.sim,
                                            plot_limit = self.plot_limit,
                                            renormalize = self.renormalize,
                                            log_g = self.log_g,
                                            overlay_probability_current = self.overlay_probability_current,
                                            distance_unit = self.distance_unit)
        self.ax_metrics = MetricsAndElectricField(self.fig.add_axes([.065, .065, .85, .2]), self.sim,
                                                  log_metrics = self.log_metrics,
                                                  metrics = self.metrics)

        self.axis_managers += [self.ax_mesh, self.ax_metrics]

        super(CylindricalSliceAnimator, self)._initialize_figure()


class PhiSliceAxis(QuantumMeshAxis):
    def initialize(self):
        unit_value, unit_name = get_unit_value_and_tex_from_unit(self.distance_unit)

        self.axis.set_theta_zero_location('N')
        self.axis.set_theta_direction('clockwise')
        self.axis.set_rlabel_position(80)

        self.axis.grid(True, color = cp.plots.COLOR_OPPOSITE_INFERNO, linestyle = ':', linewidth = 2, alpha = 0.8)  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        self.axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)  # increase size of tick labels
        self.axis.tick_params(axis = 'y', which = 'major', colors = cp.plots.COLOR_OPPOSITE_INFERNO, pad = 3)  # make r ticks a color that shows up against the colormesh

        self.axis.set_rlabel_position(80)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
        self.axis.yaxis.set_major_locator(yloc)

        plt.gcf().canvas.draw()  # must draw early to modify the axis text

        if not self.initialized:
            tick_labels = self.axis.get_yticklabels()
            for t in tick_labels:
                t.set_text(t.get_text() + r'${}$'.format(unit_name))
                self.axis.set_yticklabels(tick_labels)

        self.axis.set_rmax((self.sim.mesh.r_max - (self.sim.mesh.delta_r / 2)) / unit_value)

        self.axis.axis('tight')

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines(), *self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        super(PhiSliceAxis, self).initialize()


class SphericalSlicePhiSliceAxis(PhiSliceAxis):
    def initialize(self):
        self.mesh, self.mesh_mirror = self.sim.mesh.attach_g_to_axis(self.axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit,
                                                                     distance_unit = self.distance_unit,
                                                                     animated = True)

        self.redraw += [self.mesh, self.mesh_mirror]

        super(SphericalSlicePhiSliceAxis, self).initialize()

    def update(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)
        self.sim.mesh.update_g_mesh(self.mesh_mirror, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)

        super(SphericalSlicePhiSliceAxis, self).update()


class SphericalHarmonicPhiSliceAxis(PhiSliceAxis):
    def initialize(self):
        self.mesh = self.sim.mesh.attach_g_to_axis(self.axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, distance_unit = self.distance_unit, animated = True)

        self.redraw += [self.mesh]

        super(SphericalHarmonicPhiSliceAxis, self).initialize()

    def update(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, slicer = 'get_mesh_slicer_spatial')

        try:
            self.sim.mesh.update_probability_current_quiver(self.quiver)
        except AttributeError:
            pass

        super(SphericalHarmonicPhiSliceAxis, self).update()


class AngularMomentumDecompositionAxis(cp.AxisManager):
    def __init__(self, *args, renormalize_l_decomposition = True, **kwargs):
        self.renormalize_l_decomposition = renormalize_l_decomposition

        super(AngularMomentumDecompositionAxis, self).__init__(*args, **kwargs)

    def initialize(self):
        l_plot = self.sim.mesh.norm_by_l
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.mesh.norm()
        self.ang_mom_bar = self.axis.bar(self.sim.mesh.l, l_plot,
                                         align = 'center', color = '.5',
                                         animated = True)

        self.redraw += [*self.ang_mom_bar]

        self.axis.yaxis.grid(True, zorder = 10)

        self.axis.set_xlabel(r'Orbital Angular Momentum $\ell$', fontsize = 22)
        l_label = r'$\left| \left\langle \psi | Y^{\ell}_0 \right\rangle \right|^2$'
        if self.renormalize_l_decomposition:
            l_label += r'$/\left\langle\psi|\psi\right\rangle$'
        self.axis.set_ylabel(l_label, fontsize = 22)
        self.axis.yaxis.set_label_position('right')

        self.axis.set_ylim(0, 1)
        self.axis.set_xlim(np.min(self.sim.mesh.l) - 0.4, np.max(self.sim.mesh.l) + 0.4)

        self.axis.tick_params(labelleft = False, labelright = True)
        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)

        self.redraw += [*self.axis.yaxis.get_gridlines()]

        super(AngularMomentumDecompositionAxis, self).initialize()

    def update(self):
        l_plot = self.sim.mesh.norm_by_l
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.norm_vs_time[self.sim.data_time_index]
        for bar, height in zip(self.ang_mom_bar, l_plot):
            bar.set_height(height)

        super(AngularMomentumDecompositionAxis, self).update()


class ColorBarAxis(cp.AxisManager):
    def __init__(self, *args, colorable, **kwargs):
        self.colorable = colorable

        super(ColorBarAxis, self).__init__(*args, **kwargs)

    def initialize(self):
        self.cbar = plt.colorbar(mappable = self.colorable, cax = self.axis)
        self.cbar.ax.tick_params(labelsize = 14)

        super(ColorBarAxis, self).initialize()


class PhiSliceAnimator(WavefunctionSimulationAnimator):
    mesh_axis_type = None

    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (20, 12))

        self.ax_mesh = self.mesh_axis_type(self.fig.add_axes([.05, .05, (12 / 20) - 0.05, .9], projection = 'polar'), self.sim,
                                           plot_limit = self.plot_limit,
                                           renormalize = self.renormalize,
                                           log_g = self.log_g,
                                           overlay_probability_current = self.overlay_probability_current,
                                           distance_unit = self.distance_unit)

        legend_kwargs = {'bbox_to_anchor': (1., 1.1),
                         'loc': 'lower right',
                         'borderaxespad': 0.,
                         'fontsize': 20,
                         'fancybox': True,
                         'framealpha': .1}
        self.ax_metrics = MetricsAndElectricField(self.fig.add_axes([.575, .075, .36, .15]), self.sim,
                                                  log_metrics = self.log_metrics,
                                                  label_left = False, legend_kwargs = legend_kwargs,
                                                  metrics = self.metrics)

        self.ax_mesh.initialize()  # must pre-initialize so that the colobar can see the colormesh
        self.ax_cbar = ColorBarAxis(self.fig.add_axes([.65, .25, .03, .5]), self.sim, colorable = self.ax_mesh.mesh)

        self.axis_managers += [self.ax_mesh, self.ax_metrics, self.ax_cbar]

        plt.figtext(.075, .9, r'$|g|^2$', fontsize = 50)

        # plt.figtext(.8, .6, r'Initial State: ${}$'.format(self.spec.initial_state.tex_str), fontsize = 22)

        # self.time_text = plt.figtext(.8, .49, r'$t = {}$ as'.format(uround(self.sim.time, asec, 3)), fontsize = 30, animated = True)
        # self.redraw += [self.time_text]

        super(PhiSliceAnimator, self)._initialize_figure()

    def _update_data(self):
        super(PhiSliceAnimator, self)._update_data()

        # self.time_text.set_text(r'$t = {}$ as'.format(uround(self.sim.time, asec, 3)))


class SphericalSliceAnimator(PhiSliceAnimator):
    mesh_axis_type = SphericalSlicePhiSliceAxis


class SphericalHarmonicAnimator(PhiSliceAnimator):
    mesh_axis_type = SphericalHarmonicPhiSliceAxis

    def __init__(self, top_right_axis_manager_type = TestStateStackplot, top_right_axis_kwargs = None, **kwargs):
        self.top_right_axis_manager_type = top_right_axis_manager_type
        if top_right_axis_kwargs is None:
            top_right_axis_kwargs = {}
        self.top_right_axis_kwargs = top_right_axis_kwargs

        super(SphericalHarmonicAnimator, self).__init__(**kwargs)

    def _initialize_figure(self):
        super(SphericalHarmonicAnimator, self)._initialize_figure()

        self.top_right_axis = self.top_right_axis_manager_type(self.fig.add_axes([.56, .84, .39, .11]), self.sim, **self.top_right_axis_kwargs)

        self.axis_managers += [self.top_right_axis]
