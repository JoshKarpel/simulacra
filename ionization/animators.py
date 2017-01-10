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
    def __init__(self, axis, simulation, log_metrics = False, time_unit = 'asec', electric_field_unit = 'AEF', metrics = ('norm',)):
        self.time_unit_str = ''
        if type(time_unit) == str:
            self.time_unit_str = unit_names_to_tex_strings[time_unit]
            time_unit = unit_names_to_values[time_unit]
        self.time_unit = time_unit

        if type(electric_field_unit) == str:
            self.electric_field_unit_str = unit_names_to_tex_strings[electric_field_unit]
            self.electric_field_unit = unit_names_to_values[electric_field_unit]
        else:
            self.electric_field_unit_str = ''
            self.electric_field_unit = electric_field_unit

        self.log_metrics = log_metrics

        self.metrics = metrics

        super(MetricsAndElectricField, self).__init__(axis, simulation)

    def initialize(self):
        self.time_line, = self.axis.plot([self.sim.times[self.sim.time_index] / self.time_unit,
                                          self.sim.times[self.sim.time_index] / self.time_unit],
                                         [0, 1],
                                         color = 'gray',
                                         animated = True)

        self.redraw += [self.time_line]

        self._initialize_electric_field()

        for metric in self.metrics:
            self.__getattribute__('_initialize_metric_' + metric)()

        self.axis.grid(True, color = 'gray', linestyle = '--')

        self.axis.set_xlabel(r'Time $t$ ({})'.format(self.time_unit_str), fontsize = 24)
        self.axis.set_ylabel('Wavefunction Metric', fontsize = 24)

        self.axis.set_xlim(self.sim.times[0] / self.time_unit, self.sim.times[-1] / self.time_unit)
        self.axis.set_ylim(0, 1.025)
        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

    def _initialize_electric_field(self):
        self.axis_field = self.axis.twinx()
        self.axis_field.set_ylabel(r'Electric Field $E(t)$ ({})'.format(self.electric_field_unit_str),
                                   fontsize = 24, color = 'red')
        self.axis_field.yaxis.set_label_position('right')
        self.axis_field.tick_params(axis = 'both', which = 'major', labelsize = 14)
        self.axis_field.grid(True, color = 'red', linestyle = ':')

        for tick in self.axis_field.get_yticklabels():
            tick.set_color('red')

        self.electric_field_line, = self.axis_field.plot(self.sim.times / self.time_unit,
                                                         self.sim.electric_field_amplitude_vs_time / self.electric_field_unit,
                                                         label = r'$E(t)$ ({})'.format(self.electric_field_unit_str),
                                                         color = 'red', linewidth = 3,
                                                         animated = True)

        self.redraw += [self.electric_field_line]

    def _initialize_metric_norm(self):
        self.norm_line, = self.axis.plot(self.sim.times / self.time_unit,
                                         self.sim.norm_vs_time,
                                         label = r'$\left\langle \psi|\psi \right\rangle$',
                                         color = 'black', linewidth = 3,
                                         animated = True)

        self.redraw += [self.norm_line]

    def update(self):
        self._update_electric_field()

        for metric in self.metrics:
            self.__getattribute__('_update_metric_' + metric)()

    def _update_electric_field(self):
        self.electric_field_line.set_ydata(self.sim.electric_field_amplitude_vs_time / self.electric_field_unit)

    def _update_metric_norm(self):
        print(vars(self))
        self.norm_line.set_ydata(self.sim.norm_vs_time)


class QuantumMeshAxis(cp.AxisManager):
    def __init__(self, axis, simulation,
                 plot_limit = None,
                 renormalize = True,
                 log_g = False,
                 overlay_probability_current = False):
        self.plot_limit = plot_limit
        self.renormalize = renormalize
        self.log_g = log_g
        self.overlay_probability_current = overlay_probability_current

        super(QuantumMeshAxis, self).__init__(axis, simulation)


class QuantumMeshAnimator(cp.Animator):
    def __init__(self, *args,
                 plot_limit = None,
                 renormalize = True,
                 log_g = False,
                 log_metrics = False,
                 overlay_probability_current = False,
                 **kwargs):
        super(QuantumMeshAnimator, self).__init__(*args, **kwargs)

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


class LineAxis(QuantumMeshAxis):
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
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)

        try:
            self.sim.mesh.update_probability_current_quiver(self.quiver, plot_limit = self.plot_limit)
        except AttributeError:
            pass


class LineAnimator(QuantumMeshAnimator):
    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))

        self.ax_mesh = LineAxis(self.fig.add_axes([.06, .34, .9, .62]), self.sim,
                                plot_limit = self.plot_limit,
                                renormalize = self.renormalize,
                                log_g = self.log_g,
                                overlay_probability_current = self.overlay_probability_current)
        self.ax_metrics = MetricsAndElectricField(self.fig.add_axes([.06, .065, .9, .2]), self.sim)

        self.axis_managers += [self.ax_mesh, self.ax_metrics]

        leg = self.ax_metrics.axis.legend(loc = 'center left', fontsize = 20)  # legend must be created here so that it catches all of the lines in ax_metrics
        self.redraw += [leg]

        super(LineAnimator, self)._initialize_figure()


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

        self.axis_managers.append(CylindricalSliceAxis(self.ax_mesh))
        self.axis_managers.append(MetricsAndElectricField(self.ax_metrics))

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
        self.axis.set_theta_zero_location('N')
        self.axis.set_theta_direction('clockwise')
        self.axis.set_rlabel_position(80)

        self.axis.grid(True, color = 'silver', linestyle = ':')  # change grid color to make it show up against the colormesh

        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        self.axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)  # increase size of tick labels
        self.axis.tick_params(axis = 'y', which = 'major', colors = 'silver', pad = 3)  # make r ticks a color that shows up against the colormesh

        self.axis.axis('tight')

        self.redraw += [*self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)


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


class ColorBarAxis(cp.AxisManager):
    def __init__(self, *args, colorable, **kwargs):
        self.colorable = colorable

        super(ColorBarAxis, self).__init__(*args, **kwargs)


class PhiSliceAnimator(QuantumMeshAnimator):
    def __init__(self, *args, mesh_axis_manager_type = SphericalHarmonicPhiSlice, **kwargs):
        self.mesh_axis_manager_type = mesh_axis_manager_type

        super(PhiSliceAnimator, self).__init__(*args, **kwargs)

    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (18, 12))

        self.ax_mesh = self.fig.add_axes([.05, .05, 2 / 3 - 0.05, .9], projection = 'polar')
        self.ax_metrics = self.fig.add_axes([.6, .075, .35, .125])
        self.ax_cbar = self.fig.add_axes([.725, .275, .03, .45])

        mesh_axis = self.mesh_axis_manager_type(self.ax_mesh)
        self.axis_managers.append(mesh_axis)
        self.axis_managers.append(MetricsAndElectricField(self.ax_metrics))
        # self.axis_managers.append(ColorBarAxis(self.ax_cbar, colorable = ))

        self.ax_metrics.legend(bbox_to_anchor = (1., 1.1), loc = 'lower right', borderaxespad = 0., fontsize = 20,
                               fancybox = True, framealpha = .1)

        plt.figtext(.85, .7, r'$|g|^2$', fontsize = 50)

        plt.figtext(.8, .6, r'Initial State: ${}$'.format(self.spec.initial_state.tex_str), fontsize = 22)

        self.time_text = plt.figtext(.8, .49, r'$t = {}$ as'.format(uround(self.sim.time, asec, 3)), fontsize = 30, animated = True)
        self.redraw += [self.time_text]

    def _update_data(self):
        super(PhiSliceAnimator, self)._update_data()

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


class SphericalSliceAnimator(PhiSliceAnimator):
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


class SphericalHarmonicAnimator(PhiSliceAnimator):
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
