import functools
import logging
import os
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import compy as cp
from compy.units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Animator:
    """
    A class that handles sending frames to ffmpeg to create animations.

    Should be subclassed and the hook methods _initialize_figure and update_frame should be overridden/extended.

    ffmpeg must be visible on the system path.
    """

    def __init__(self, postfix = '', target_dir = None,
                 length = 30, fps = 30,
                 plot_limit = None, renormalize = True, log = False, overlay_probability_current = False,
                 colormap = plt.cm.inferno):
        if target_dir is None:
            target_dir = os.getcwd()
        self.target_dir = target_dir

        postfix = cp.utils.strip_illegal_characters(postfix)
        if postfix != '' and not postfix.startswith('_'):
            postfix = '_' + postfix
        self.postfix = postfix

        self.length = length
        self.fps = fps
        self.plot_limit = plot_limit
        self.renormalize = renormalize
        self.log = log
        self.overlay_probability_current = overlay_probability_current
        self.colormap = colormap

        self.sim = None
        self.spec = None
        self.fig = None

    def initialize(self, simulation):
        """Hook for second part of initialization, once the Simulation is known."""
        self.sim = simulation
        self.spec = simulation.spec

        self.file_name = '{}{}.mp4'.format(self.sim.file_name, self.postfix)
        self.file_path = os.path.join(self.target_dir, self.file_name)
        cp.utils.ensure_dir_exists(self.file_path)
        try:
            os.remove(self.file_path)
        except FileNotFoundError:
            pass

        ideal_frame_count = self.length * self.fps
        self.decimation = int(self.sim.time_steps / ideal_frame_count)
        if self.decimation < 1:
            self.decimation = 1
        self.fps = (self.sim.time_steps / self.decimation) / self.length

        self._initialize_figure()

        canvas_width, canvas_height = self.fig.canvas.get_width_height()
        self.cmdstring = ("ffmpeg",
                          '-y',
                          '-r', '{}'.format(self.fps),  # choose fps
                          '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                          '-pix_fmt', 'argb',  # format
                          '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                          '-vcodec', 'mpeg4',  # output encoding
                          '-q:v', '1',
                          self.file_path)

        self.ffmpeg = subprocess.Popen(self.cmdstring, stdin = subprocess.PIPE, bufsize = -1)

        logger.info('Initialized {}'.format(self, self.sim))

    def cleanup(self):
        self.ffmpeg.communicate()
        logger.info('Cleaned up {}'.format(self, self.sim))

    def _initialize_figure(self):
        logger.debug('Initialized figure for {}'.format(self))

    def _update_frame(self):
        logger.debug('{} updated frame from {}'.format(self.__class__.__name__, self.sim.name))

    def send_frame_to_ffmpeg(self):
        self._update_frame()

        self.fig.canvas.draw()
        string = self.fig.canvas.tostring_argb()

        self.ffmpeg.stdin.write(string)

        logger.debug('{} sent frame to ffpmeg from {}'.format(self.__class__.__name__, self.sim.name))

    def __str__(self):
        try:
            lim = str(uround(self.plot_limit, bohr_radius, 3)) + ' Bohr radii'
        except TypeError:
            lim = None

        return '{}(plot limit = {}, renormalize = {}, log = {}, probability current = {})'.format(self.__class__.__name__,
                                                                                                  lim,
                                                                                                  self.renormalize,
                                                                                                  self.log,
                                                                                                  self.overlay_probability_current,
                                                                                                  self.sim)

    def __repr__(self):
        return '{}(length = {}, fps = [}, plot_limit = {}, renormalize = {}, log = {}, overlay_probability_current = {})'.format(self.__class__.__name__,
                                                                                                                                 self.length,
                                                                                                                                 self.fps,
                                                                                                                                 self.plot_limit,
                                                                                                                                 self.renormalize,
                                                                                                                                 self.log,
                                                                                                                                 self.overlay_probability_current,
                                                                                                                                 self.sim)


class CylindricalSliceAnimator(Animator):
    def initialize(self, simulation):
        Animator.initialize(self, simulation)
        self.ax_time.legend(loc = 'center left', fontsize = 20)  # legend must be created here so that it catches all of the lines in ax_time

    def _initialize_figure(self):
        plt.set_cmap(self.colormap)

        self.fig = plt.figure(figsize = (16, 12))

        self.ax_mesh = self.fig.add_axes([.06, .34, .9, .62])
        self.ax_time = self.fig.add_axes([.06, .065, .9, .2])

        self._initialize_mesh_axis()
        self._initialize_time_axis()

    def _initialize_mesh_axis(self):
        self.mesh = self.sim.mesh.attach_g_to_axis(self.ax_mesh, normalize = self.renormalize, log = self.log, plot_limit = self.plot_limit)

        if self.overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(self.ax_mesh, plot_limit = self.plot_limit)

        self.ax_mesh.grid(True, color = 'silver', linestyle = ':')  # change grid color to make it show up against the colormesh
        self.ax_time.grid(True)

        self.ax_mesh.set_xlabel(r'$z$ (Bohr radii)', fontsize = 24)
        self.ax_mesh.set_ylabel(r'$\rho$ (Bohr radii)', fontsize = 24)
        self.ax_time.set_xlabel('Time $t$ (as)', fontsize = 24)
        self.ax_time.set_ylabel('Ionization Metric', fontsize = 24)

        self.ax_time.set_xlim(self.sim.times[0] / asec, self.sim.times[-1] / asec)
        self.ax_time.set_ylim(0, 1)

        self.ax_time.tick_params(labelright = True)
        self.ax_time.tick_params(axis = 'both', which = 'major', labelsize = 14)
        self.ax_mesh.tick_params(axis = 'both', which = 'major', labelsize = 14)

        divider = make_axes_locatable(self.ax_mesh)
        cax = divider.append_axes("right", size = "2%", pad = 0.05)
        self.cbar = plt.colorbar(cax = cax, mappable = self.mesh)
        self.cbar.ax.tick_params(labelsize = 14)

        self.ax_mesh.axis('tight')

    def _initialize_time_axis(self):
        if self.spec.electric_potential is not None:
            self.field_max = np.abs(np.max(self.spec.electric_potential.get_amplitude(self.sim.times)))
            self.electric_field_line, = self.ax_time.plot(self.sim.times / asec, np.abs(self.sim.electric_field_amplitude_vs_time) / self.field_max,
                                                          label = r'$|E|/\left|E_{\mathrm{max}}\right|$',
                                                          color = 'red', linewidth = 2)

        self.norm_line, = self.ax_time.plot(self.sim.times / asec, self.sim.norm_vs_time,
                                            label = r'$\left\langle \psi|\psi \right\rangle$',
                                            color = 'black', linestyle = '--', linewidth = 3)

        self.overlaps_stackplot = self.ax_time.stackplot(self.sim.times / asec, *self._compute_stackplot_overlaps(),
                                                         labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$',
                                                                   r'$\left| \left\langle \psi|\psi_{n\leq5} \right\rangle \right|^2$'],
                                                         colors = ['.3', '.5'])

        self.time_line, = self.ax_time.plot([self.sim.times[self.sim.time_index] / asec, self.sim.times[self.sim.time_index] / asec], [0, 1],
                                            linestyle = '-.', color = 'gray')

    def _compute_stackplot_overlaps(self):
        initial_overlap = [self.sim.state_overlaps_vs_time[self.spec.initial_state]]
        non_initial_overlaps = [self.sim.state_overlaps_vs_time[state] for state in self.spec.test_states if state != self.spec.initial_state]
        total_non_initial_overlaps = functools.reduce(np.add, non_initial_overlaps)
        overlaps = [initial_overlap, total_non_initial_overlaps]

        return overlaps

    def _update_frame(self):
        plt.set_cmap(self.colormap)
        self._update_mesh_axis()
        self._update_time_axis()
        super(CylindricalSliceAnimator, self)._update_frame()

    def _update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log, plot_limit = self.plot_limit)

        if self.overlay_probability_current:
            self.sim.mesh.update_probability_current_quiver(self.quiver, plot_limit = self.plot_limit)

    def _update_time_axis(self):
        try:
            self.electric_field_line.set_ydata(np.abs(self.sim.electric_field_amplitude_vs_time) / self.field_max)
        except AttributeError:
            pass

        self.norm_line.set_ydata(self.sim.norm_vs_time)
        self.overlaps_stackplot = self.ax_time.stackplot(self.sim.times / asec, *self._compute_stackplot_overlaps(),
                                                         labels = ['Initial State Overlap', r'Overlap with $n \leq 5$'], colors = ['.3', '.5'])
        self.time_line.set_xdata([self.sim.times[self.sim.time_index] / asec, self.sim.times[self.sim.time_index] / asec])


class SphericalSliceAnimator(CylindricalSliceAnimator):
    def initialize(self, simulation):
        Animator.initialize(self, simulation)
        legend = self.ax_time.legend(bbox_to_anchor = (1., 1.1), loc = 'lower right', borderaxespad = 0., fontsize = 20,
                                     fancybox = True, framealpha = 0)
        # legend must be created here so that it catches all of the lines in ax_time
        # TODO: is this really still true?

    def _initialize_figure(self):
        plt.set_cmap(self.colormap)

        self.fig = plt.figure(figsize = (18, 12))

        self.ax_mesh = self.fig.add_axes([.05, .05, 2 / 3 - 0.05, .9], projection = 'polar')
        self.ax_time = self.fig.add_axes([.6, .075, .35, .125])
        self.cbar_axis = self.fig.add_axes([.725, .275, .03, .675])

        plt.figtext(.62, .87, r'$|g|^2$', fontsize = 50)

        plt.figtext(.8, .9, r'Simulation:', fontsize = 22)
        plt.figtext(.82, .85, self.sim.name, fontsize = 20)
        plt.figtext(.8, .8, r'Initial State: ${}$'.format(self.spec.initial_state.tex_str), fontsize = 22)
        # TODO: time text?
        # TODO: replace with a for over a given list of strings

        self._mesh_setup()

        self.ax_mesh.set_theta_zero_location('N')
        self.ax_mesh.set_theta_direction('clockwise')

        if self.overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(self.ax_mesh)

        if self.spec.electric_potential is not None:
            self.field_max = np.abs(np.max(self.spec.electric_potential.get_amplitude(self.sim.times)))
            self.electric_field_line, = self.ax_time.plot(self.sim.times / asec, np.abs(self.sim.electric_field_amplitude_vs_time) / self.field_max,
                                                          label = r'$|E|/\left|E_{\mathrm{max}}\right|$', color = 'red', linewidth = 2)

        self.norm_line, = self.ax_time.plot(self.sim.times / asec, self.sim.norm_vs_time,
                                            label = r'$\left\langle \psi|\psi \right\rangle$',
                                            color = 'black', linestyle = '--', linewidth = 3)

        self.overlaps_stackplot = self.ax_time.stackplot(self.sim.times / asec, *self._compute_stackplot_overlaps(),
                                                         labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$',
                                                                   r'$\left| \left\langle \psi|\psi_{n\leq5} \right\rangle \right|^2$'],
                                                         colors = ['.3', '.5'])

        self.time_line, = self.ax_time.plot([self.sim.times[self.sim.time_index] / asec, self.sim.times[self.sim.time_index] / asec], [0, 1],
                                            linestyle = '-.', color = 'gray')

        self.ax_mesh.grid(True, color = 'silver', linestyle = ':')  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        # angle_labels = ['\u03b8=0\u00b0'] + angle_labels
        self.ax_mesh.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)
        self.ax_time.grid()

        self.ax_time.set_xlabel('Time (as)', fontsize = 22)
        self.ax_time.set_ylabel('Ionization Metric', fontsize = 22)
        self.ax_time.yaxis.set_label_position('right')

        self.ax_time.set_xlim(self.sim.times[0] / asec, self.sim.times[-1] / asec)
        self.ax_time.set_ylim(0, 1)

        self.ax_mesh.tick_params(axis = 'both', which = 'major', labelsize = 20)  # increase size of tick labels
        self.ax_mesh.tick_params(axis = 'y', which = 'major', colors = 'silver', pad = 3)  # make r ticks a color that shows up against the colormesh
        self.ax_mesh.tick_params(axis = 'both', which = 'both', length = 0)

        self.ax_mesh.set_rlabel_position(80)
        # last_r_label = self.ax_mesh.get_yticklabels()[-1]
        # last_r_label.set_color('black')  # last r tick is outside the colormesh, so make it black again

        self.ax_time.tick_params(labelleft = False, labelright = True)
        self.ax_time.tick_params(axis = 'both', which = 'major', labelsize = 14)

        self.cbar = plt.colorbar(mappable = self.mesh, cax = self.cbar_axis)
        self.cbar.ax.tick_params(labelsize = 14)

        self.ax_mesh.axis('tight')
        # self.ax_time.axis('tight')

        if self.plot_limit is None:
            self.ax_mesh.set_rmax((self.sim.mesh.r_max - (self.sim.mesh.delta_r / 2)) / bohr_radius)
        else:
            self.ax_mesh.set_rmax(self.plot_limit / bohr_radius)

    def _mesh_setup(self):
        self.mesh, self.mesh_mirror = self.sim.mesh.attach_g_to_axis(self.ax_mesh, normalize = self.renormalize, log = self.log)

    def _update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log)
        self.sim.mesh.update_g_mesh(self.mesh_mirror, normalize = self.renormalize, log = self.log)

        if self.overlay_probability_current:
            self.sim.mesh.update_probability_current_quiver(self.quiver)


class SphericalHarmonicAnimator(SphericalSliceAnimator):
    def _mesh_setup(self):
        self.mesh = self.sim.mesh.attach_g_to_axis(self.ax_mesh, normalize = self.renormalize, log = self.log)

    def _update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.renormalize, log = self.log)

        if self.overlay_probability_current:
            self.sim.mesh.update_probability_current_quiver(self.quiver)
