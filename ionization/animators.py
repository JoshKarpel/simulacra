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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MetricsAndElectricField(cp.AxisManager):
    def __init__(self, *args, time_unit = 'asec', metrics = ('norm',), **kwargs):
        super(MetricsAndElectricField, self).__init__(*args, **kwargs)

        if type(time_unit) == str:
            self.time_unit_str = unit_names_to_tex_strings[time_unit]
            self.time_unit = unit_names_to_values[time_unit]
        else:
            self.time_unit_str = ''
            self.time_unit = time_unit

        self.metrics = metrics

    def initialize(self):
        self.time_line, = self.axis.plot([self.sim.times[self.sim.time_index] / asec, self.sim.times[self.sim.time_index] / asec], [0, 1],
                                         color = 'gray',
                                         animated = True)

        for metric in self.metrics:
            self.__getattribute__('_initialize_metric_' + metric)

        self.axis.grid(True)

        self.axis.set_xlabel('Time $t$ (as)', fontsize = 24)
        self.axis.set_ylabel('Ionization Metric', fontsize = 24)

        self.axis.set_xlim(self.sim.times[0] / asec, self.sim.times[-1] / asec)
        self.axis.set_ylim(-.01, 1.01)

        # self.axis.tick_params(labelleft = left_tick_labels, labelright = right_tick_labels)
        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        # if label_right:
        #     self.axis.yaxis.set_label_position('right')

    def _initialize_metric_norm(self):
        self.norm_line, = self.axis.plot(self.sim.times / asec, self.sim.norm_vs_time,
                                         label = r'$\left\langle \psi|\psi \right\rangle$',
                                         color = 'black', linewidth = 3,
                                         animated = True)

    def _initialize_metric_initial_state_overlap(self):
        raise NotImplementedError

    def _initialize_metric_test_state_overlaps(self):
        self.overlaps_stackplot = self.axis.stackplot(self.sim.times / asec, *self._compute_stackplot_overlaps(),
                                                      labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$',
                                                                r'$\left| \left\langle \psi|\psi_{{n \leq {}}} \right\rangle \right|^2$'.format(max(self.sim.spec.test_states, key = lambda s: s.n).n)],
                                                      colors = ['.3', '.5'],
                                                      animated = True)

    def _initialize_metric_total_test_state_overlaps(self):
        raise NotImplementedError

    def update(self):
        for metric in self.metrics:
            self.__getattribute__('_update_metric_' + metric)()

    def _update_metric_norm(self):
        self.norm_line.set_ydata(self.sim.norm_vs_time)

    def _update_metric_initial_state_overlap(self):
        raise NotImplementedError

    def _update_metric_test_state_overlaps(self):
        raise NotImplementedError

    def _update_metric_total_test_state_overlaps(self):
        raise NotImplementedError


class QuantumMeshAxis(cp.AxisManager):
    def __init__(self, renormalize = True):
        self.renormalize = renormalize


class QuantumMeshAnimator(cp.Animator):
    def __init__(self, plot_limit = None, renormalize = True, log_g = False, log_metrics = False, overlay_probability_current = False, **kwargs):
        super(QuantumMeshAnimator, self).__init__(**kwargs)

        self.plot_limit = plot_limit
        self.renormalize = renormalize
        self.log_g = log_g
        self.log_metrics = log_metrics
        self.overlay_probability_current = overlay_probability_current

    def __str__(self):
        try:
            lim = str(uround(self.plot_limit, bohr_radius, 3)) + ' Bohr radii'
        except TypeError:
            lim = None

        return '{}(postfix = "{}", plot limit = {}, renormalize = {}, log = {}, probability current = {})'.format(self.__class__.__name__,
                                                                                                                  self.postfix,
                                                                                                                  lim,
                                                                                                                  self.renormalize,
                                                                                                                  self.log_g,
                                                                                                                  self.overlay_probability_current,
                                                                                                                  self.sim)

    def __repr__(self):
        return '{}(postfix = {}, length = {}, fps = [}, plot_limit = {}, renormalize = {}, log = {}, overlay_probability_current = {})'.format(self.__class__.__name__,
                                                                                                                                               self.postfix,
                                                                                                                                               self.length,
                                                                                                                                               self.fps,
                                                                                                                                               self.plot_limit,
                                                                                                                                               self.renormalize,
                                                                                                                                               self.log_g,
                                                                                                                                               self.overlay_probability_current,
                                                                                                                                               self.sim)

    # def _update_data(self):
    #     self._update_metric_axis()
    #     self._update_mesh_axis()
    #
    #     super(QuantumMeshAnimator, self)._update_data()

    def _make_metrics_axis(self, axis, label_right = False, left_tick_labels = True, right_tick_labels = True):
        """
        Make the given axis an axis that shows wavefunction metrics vs. time as the animation proceeds.

        :param axis: the axis to make into a metrics plot
        :param label_right: if False, y-axis label will be on the left. If True, on the right. Defaults to False.
        :param left_tick_labels: draw tick labels on the left (defaults to True)
        :param right_tick_labels: draw tick labels on the right (defaults to True)
        :return: the axis
        """
        self.overlaps_stackplot = axis.stackplot(self.sim.times / asec, *self._compute_stackplot_overlaps(),
                                                 labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$',
                                                           r'$\left| \left\langle \psi|\psi_{{n \leq {}}} \right\rangle \right|^2$'.format(max(self.sim.spec.test_states, key = lambda s: s.n).n)],
                                                 colors = ['.3', '.5'],
                                                 animated = True)

        self.norm_line, = axis.plot(self.sim.times / asec, self.sim.norm_vs_time,
                                    label = r'$\left\langle \psi|\psi \right\rangle$',
                                    color = 'black', linewidth = 3,
                                    animated = True)

        self.time_line, = axis.plot([self.sim.times[self.sim.time_index] / asec, self.sim.times[self.sim.time_index] / asec], [0, 1],
                                    color = 'gray',
                                    animated = True)

        self.redraw += [*self.overlaps_stackplot, self.norm_line, self.time_line]

        if self.spec.electric_potential is not None:
            self.field_max = np.abs(np.max(self.spec.electric_potential.get_electric_field_amplitude(self.sim.times)))
            self.electric_field_line, = axis.plot(self.sim.times / asec, np.abs(self.sim.electric_field_amplitude_vs_time) / self.field_max,
                                                  label = r'$|E|/\left|E_{\mathrm{max}}\right|$',
                                                  color = 'red', linewidth = 2,
                                                  animated = True)

            self.redraw += [self.electric_field_line]

        axis.grid(True)

        axis.set_xlabel('Time $t$ (as)', fontsize = 24)
        axis.set_ylabel('Ionization Metric', fontsize = 24)

        axis.set_xlim(self.sim.times[0] / asec, self.sim.times[-1] / asec)
        axis.set_ylim(-.01, 1.01)

        axis.tick_params(labelleft = left_tick_labels, labelright = right_tick_labels)
        axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        if label_right:
            axis.yaxis.set_label_position('right')

        return axis

    def _update_metric_axis(self):
        # redrawing the stackplot is a huge disaster, because it needs to be fully recreated, so all the reference in self.redraw need to be removed and then added back in
        self.redraw = [rd for rd in self.redraw if rd not in self.overlaps_stackplot]
        self.overlaps_stackplot = self.ax_metrics.stackplot(self.sim.times / asec, *self._compute_stackplot_overlaps(),
                                                            labels = ['Initial State Overlap', r'Overlap with $n \leq 5$'], colors = ['.3', '.5'])
        self.redraw = [*self.overlaps_stackplot] + self.redraw  # this hack is awful, but keeps the new stackplot artists always as the first thing redrawn

        self.norm_line.set_ydata(self.sim.norm_vs_time)

        try:
            self.electric_field_line.set_ydata(np.abs(self.sim.electric_field_amplitude_vs_time) / self.field_max)
        except AttributeError:
            pass

        self.time_line.set_xdata([self.sim.times[self.sim.time_index] / asec, self.sim.times[self.sim.time_index] / asec])

    def _compute_stackplot_overlaps(self):
        initial_overlap = [self.sim.state_overlaps_vs_time[self.spec.initial_state]]
        non_initial_overlaps = [self.sim.state_overlaps_vs_time[state] for state in self.spec.test_states if state != self.spec.initial_state]
        total_non_initial_overlaps = functools.reduce(np.add, non_initial_overlaps)
        overlaps = [initial_overlap, total_non_initial_overlaps]

        return overlaps

    def _make_mesh_axis(self, axis):
        raise NotImplementedError

    def _update_mesh_axis(self):
        raise NotImplementedError


class LineAxis(cp.AxisManager):
    def initialize(self):
        self.mesh = self.sim.mesh.attach_g_to_axis(self.axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, animated = True)
        self.redraw += [self.mesh]

        if self.overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(self.axis, plot_limit = self.plot_limit, animated = True)
            self.redraw += [self.quiver]

        self.axis.grid(True, color = 'silver', linestyle = ':')  # change grid color to make it show up against the colormesh

        self.axis.set_xlabel(r'$x$ (nm)', fontsize = 24)
        self.axis.set_ylabel(r'$\left|g\right|^2$', fontsize = 24)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        self.axis.axis('tight')

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

    def update(self):
        raise NotImplementedError


class LineAnimator(QuantumMeshAnimator):
    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))

        self.ax_mesh = self.fig.add_axes([.06, .34, .9, .62])
        self.ax_metrics = self.fig.add_axes([.06, .065, .9, .2])

        self._make_mesh_axis(self.ax_mesh)
        self._make_metrics_axis(self.ax_metrics)

        leg = self.ax_metrics.legend(loc = 'center left', fontsize = 20)  # legend must be created here so that it catches all of the lines in ax_metrics
        self.redraw += [leg]

        super(LineAnimator, self)._initialize_figure()

    def _make_mesh_axis(self, axis):
        """
        Make the given axis an axis that shows the wavefunction on the mesh as the animation proceeds.

        :param axis: the axis to make into a mesh display
        :return: the axis
        """
        self.mesh = self.sim.mesh.attach_g_to_axis(axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, animated = True)
        self.redraw += [self.mesh]

        if self.overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(axis, plot_limit = self.plot_limit, animated = True)
            self.redraw += [self.quiver]

        axis.grid(True, color = 'silver', linestyle = ':')  # change grid color to make it show up against the colormesh

        axis.set_xlabel(r'$x$ (nm)', fontsize = 24)
        axis.set_ylabel(r'$\left|g\right|^2$', fontsize = 24)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        axis.axis('tight')

        self.redraw += [*axis.xaxis.get_gridlines(), *axis.yaxis.get_gridlines()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        return axis

    def _update_mesh_axis(self):  # TODO: factor out
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)

        try:
            self.sim.mesh.update_probability_current_quiver(self.quiver, plot_limit = self.plot_limit)
        except AttributeError:
            pass


class CylindricalSliceAxis(cp.AxisManager):
    def draw(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class CylindricalSliceAnimator(QuantumMeshAnimator):
    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))

        self.ax_mesh = self.fig.add_axes([.06, .34, .9, .62])
        self.ax_metrics = self.fig.add_axes([.06, .065, .9, .2])

        self._make_mesh_axis(self.ax_mesh)
        self._make_metrics_axis(self.ax_metrics)

        leg = self.ax_metrics.legend(loc = 'center left', fontsize = 20)  # legend must be created here so that it catches all of the lines in ax_metrics
        self.redraw += [leg]

        super(CylindricalSliceAnimator, self)._initialize_figure()

    def _make_mesh_axis(self, axis):
        """
        Make the given axis an axis that shows the wavefunction on the mesh as the animation proceeds.

        :param axis: the axis to make into a mesh display
        :return: the axis
        """
        self.mesh = self.sim.mesh.attach_g_to_axis(axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, animated = True)
        self.redraw += [self.mesh]

        if self.overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(axis, plot_limit = self.plot_limit, animated = True)
            self.redraw += [self.quiver]

        axis.grid(True, color = 'silver', linestyle = ':')  # change grid color to make it show up against the colormesh

        axis.set_xlabel(r'$z$ (Bohr radii)', fontsize = 24)
        axis.set_ylabel(r'$\rho$ (Bohr radii)', fontsize = 24)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        divider = make_axes_locatable(self.ax_mesh)
        cax = divider.append_axes("right", size = "2%", pad = 0.05)
        self.cbar = plt.colorbar(cax = cax, mappable = self.mesh)
        self.cbar.ax.tick_params(labelsize = 14)

        axis.axis('tight')

        self.redraw += [*axis.xaxis.get_gridlines(), *axis.yaxis.get_gridlines()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        return axis

    def _update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)

        try:
            self.sim.mesh.update_probability_current_quiver(self.quiver, plot_limit = self.plot_limit)
        except AttributeError:
            pass


class PhiSlice(cp.AxisManager):
    def initialize(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class SphericalSlicePhiSlice(PhiSlice):
    def initialize(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class SphericalHarmonicPhiSlice(PhiSlice):
    def initialize(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class AngularMomentumDecomposition(cp.AxisManager):
    def initialize(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class PolarAnimator(QuantumMeshAnimator):
    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (18, 12))

        self.ax_mesh = self.fig.add_axes([.05, .05, 2 / 3 - 0.05, .9], projection = 'polar')
        self.ax_metrics = self.fig.add_axes([.6, .075, .35, .125])
        self.cbar_axis = self.fig.add_axes([.725, .275, .03, .45])

        self._make_mesh_axis(self.ax_mesh)
        self._make_metrics_axis(self.ax_metrics, label_right = True, left_tick_labels = False)
        self._make_cbar_axis(self.cbar_axis)

        self.ax_metrics.legend(bbox_to_anchor = (1., 1.1), loc = 'lower right', borderaxespad = 0., fontsize = 20,
                               fancybox = True, framealpha = .1)

        plt.figtext(.85, .7, r'$|g|^2$', fontsize = 50)

        plt.figtext(.8, .6, r'Initial State: ${}$'.format(self.spec.initial_state.tex_str), fontsize = 22)

        self.time_text = plt.figtext(.8, .49, r'$t = {}$ as'.format(uround(self.sim.time, asec, 3)), fontsize = 30, animated = True)
        self.redraw += [self.time_text]

    def _update_data(self):
        super(PolarAnimator, self)._update_data()

        self.time_text.set_text(r'$t = {}$ as'.format(uround(self.sim.time, asec, 3)))

    def _make_meshes(self, axis):
        raise NotImplementedError

    def _make_mesh_axis(self, axis):
        self._make_meshes(axis)

        axis.set_theta_zero_location('N')
        axis.set_theta_direction('clockwise')
        axis.set_rlabel_position(80)

        axis.grid(True, color = 'silver', linestyle = ':')  # change grid color to make it show up against the colormesh

        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 20)  # increase size of tick labels
        axis.tick_params(axis = 'y', which = 'major', colors = 'silver', pad = 3)  # make r ticks a color that shows up against the colormesh

        axis.axis('tight')

        self.redraw += [*axis.xaxis.get_gridlines(), *axis.yaxis.get_gridlines(), *axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

    def _make_cbar_axis(self, axis):
        self.cbar = plt.colorbar(mappable = self.mesh, cax = axis)
        self.cbar.ax.tick_params(labelsize = 14)


class SphericalSliceAnimator(PolarAnimator):
    def _make_meshes(self, axis):
        self.mesh, self.mesh_mirror = self.sim.mesh.attach_g_to_axis(axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit,
                                                                     animated = True)

        self.redraw += [self.mesh, self.mesh_mirror]

    def _update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)
        self.sim.mesh.update_g_mesh(self.mesh_mirror, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)

        try:
            self.sim.mesh.update_probability_current_quiver(self.quiver)
        except AttributeError:
            pass


class SphericalHarmonicAnimator(PolarAnimator):
    def __init__(self, renormalize_l_decomposition = True, **kwargs):
        super(SphericalHarmonicAnimator, self).__init__(**kwargs)

        self.renormalize_l_decomposition = renormalize_l_decomposition

    def _initialize_figure(self):
        super(SphericalHarmonicAnimator, self)._initialize_figure()

        self.ax_ang_mom = self.fig.add_axes([.6, .84, .345, .125])

        self._make_ang_mom_axis(self.ax_ang_mom)

    def _update_data(self):
        super(SphericalHarmonicAnimator, self)._update_data()

        self._update_ang_mom_axis()

    def _make_meshes(self, axis):
        self.mesh = self.sim.mesh.attach_g_to_axis(axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, animated = True)

        self.redraw += [self.mesh]

    def _update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit, slicer = 'get_mesh_slicer_spatial')

        try:
            self.sim.mesh.update_probability_current_quiver(self.quiver)
        except AttributeError:
            pass

    def _make_ang_mom_axis(self, axis):
        """
        Make the given axis an axis that shows the probability to be in each angular momentum state as the animation proceeds.

        :param axis: the axis to make into an angular momentum display
        :return: the axis
        """
        l_plot = self.sim.mesh.norm_by_l
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.mesh.norm
        self.ang_mom_bar = axis.bar(self.sim.mesh.l, l_plot,
                                    align = 'center', color = '.5',
                                    animated = True)

        self.redraw += [*self.ang_mom_bar]

        axis.yaxis.grid(True, zorder = 10)

        axis.set_xlabel(r'Orbital Angular Momentum $\ell$', fontsize = 22)
        l_label = r'$\left| \left\langle \psi | Y^{\ell}_0 \right\rangle \right|^2$'
        if self.renormalize_l_decomposition:
            l_label += r'$/\left\langle\psi|\psi\right\rangle$'
        axis.set_ylabel(l_label, fontsize = 22)
        axis.yaxis.set_label_position('right')

        axis.set_ylim(0, 1)
        axis.set_xlim(np.min(self.sim.mesh.l) - 0.4, np.max(self.sim.mesh.l) + 0.4)

        axis.tick_params(labelleft = False, labelright = True)
        axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        return axis

    def _update_ang_mom_axis(self):
        l_plot = self.sim.mesh.norm_by_l
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.norm_vs_time[self.sim.time_index]
        for bar, height in zip(self.ang_mom_bar, l_plot):
            bar.set_height(height)
