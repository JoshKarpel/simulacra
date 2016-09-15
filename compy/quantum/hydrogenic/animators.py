import logging
import os
import functools
import subprocess

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from compy import utils
import compy.units as un

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CylindricalSliceAnimator:
    def __init__(self, simulation, cluster = False):
        self.sim = simulation
        self.spec = simulation.spec
        self.cluster = cluster

        plt.set_cmap(plt.cm.viridis)
        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['ytick.major.pad'] = 5

        ideal_frame_count = self.spec.animation_time * self.spec.animation_fps
        self.animation_decimation = int(self.sim.time_steps / ideal_frame_count)
        if self.animation_decimation < 1:
            self.animation_decimation = 1
        self.fps = (self.sim.time_steps / self.animation_decimation) / self.spec.animation_time

        if cluster:
            self.filename = self.sim.spec.pickle_name + '.mp4'
        else:
            if self.spec.animation_dir is not None:
                target_dir = self.spec.animation_dir
            else:
                target_dir = os.getcwd()

            postfix = ''
            if self.spec.animation_log_g:
                postfix += '_log'

            self.filename = os.path.join(target_dir, '{}{}.mp4'.format(self.spec.name, postfix))

        utils.ensure_dir_exists(self.filename)

        try:
            os.remove(self.filename)
        except FileNotFoundError:
            pass

        self.cmdstring = None
        self.ffmpeg = None

    def initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))
        self.fig.set_tight_layout(True)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [3, 1])
        self.ax_mesh = plt.subplot(grid_spec[0])
        self.ax_time = plt.subplot(grid_spec[1])

        self.ax_mesh.grid()
        self.ax_time.grid()

        self.ax_mesh.set_xlabel(r'$z$ (Bohr radii)', fontsize = 18)
        self.ax_mesh.set_ylabel(r'$\rho$ (Bohr radii)', fontsize = 18)
        self.ax_time.set_xlabel('Time (as)', fontsize = 18)
        self.ax_time.set_ylabel('Ionization Metric', fontsize = 18)

        self.ax_time.set_xlim(self.sim.times[0] / un.asec, self.sim.times[-1] / un.asec)
        self.ax_time.set_ylim(0, 1)

        self.ax_time.tick_params(labelright = True)
        self.ax_time.tick_params(axis = 'both', which = 'major', labelsize = 12)
        self.ax_mesh.tick_params(axis = 'both', which = 'major', labelsize = 12)

        self.mesh = self.sim.mesh.attach_g_to_axis(self.ax_mesh, normalize = self.spec.animation_normalize, log = self.spec.animation_log_g, plot_limit = self.spec.animation_plot_limit)

        divider = make_axes_locatable(self.ax_mesh)
        cax = divider.append_axes("right", size = "2%", pad = 0.05)
        self.cbar = plt.colorbar(cax = cax, mappable = self.mesh)
        self.cbar.ax.tick_params(labelsize = 12)

        if self.spec.animation_overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(self.ax_mesh, plot_limit = self.spec.animation_plot_limit)

        if self.spec.electric_potential is not None:
            self.pulse_max = np.max(self.spec.electric_potential.get_amplitude(self.sim.times))
            self.electric_field_line, = self.ax_time.plot(self.sim.times / un.asec, np.abs(self.sim.electric_field_amplitude_vs_time / self.pulse_max),
                                                          label = r'Normalized $|E|$',
                                                          color = 'red', linewidth = 2)

        self.norm_line, = self.ax_time.plot(self.sim.times / un.asec, self.sim.norm_vs_time,
                                            label = r'$\left\langle \psi|\psi \right\rangle$',
                                            color = 'black', linestyle = '--', linewidth = 3)

        self.overlaps_stackplot = self.ax_time.stackplot(self.sim.times / un.asec, *self.compute_stackplot_overlaps(),
                                                         labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$',
                                                                   r'$\left| \left\langle \psi|\psi_{n\leq5} \right\rangle \right|^2$'],
                                                         colors = ['.3', '.5'])

        self.time_line, = self.ax_time.plot([self.sim.times[self.sim.time_index] / un.asec, self.sim.times[self.sim.time_index] / un.asec], [0, 1],
                                            linestyle = '-.', color = 'gray')

        self.ax_mesh.axis('tight')

    def compute_stackplot_overlaps(self):
        initial_overlap = [self.sim.state_overlaps_vs_time[self.spec.initial_state]]
        non_initial_overlaps = [self.sim.state_overlaps_vs_time[state] for state in self.spec.test_states if state != self.spec.initial_state]
        total_non_initial_overlaps = functools.reduce(np.add, non_initial_overlaps)
        overlaps = [initial_overlap, total_non_initial_overlaps]

        return overlaps

    def update_frame(self):
        self.update_mesh_axis()
        self.update_time_axis()

    def update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.spec.animation_normalize, log = self.spec.animation_log_g, plot_limit = self.spec.animation_plot_limit)

        if self.spec.animation_overlay_probability_current:
            self.sim.mesh.update_probability_current_quiver(self.quiver, plot_limit = self.spec.animation_plot_limit)

    def update_time_axis(self):
        try:
            self.electric_field_line.set_ydata(np.abs(self.sim.external_potential_amplitude_vs_time / self.pulse_max))
        except AttributeError:
            pass

        self.norm_line.set_ydata(self.sim.norm_vs_time)
        self.overlaps_stackplot = self.ax_time.stackplot(self.sim.times / un.asec, *self.compute_stackplot_overlaps(),
                                                         labels = ['Initial State Overlap', r'Overlap with $n \leq 5$'], colors = ['.3', '.5'])
        self.time_line.set_xdata([self.sim.times[self.sim.time_index] / un.asec, self.sim.times[self.sim.time_index] / un.asec])

    def initialize(self):
        self.initialize_figure()

        self.ax_time.legend(loc = 'center left', fontsize = 14)  # legend must be created here so that it catches all of the lines in ax_time

        canvas_width, canvas_height = self.fig.canvas.get_width_height()

        self.cmdstring = ("ffmpeg",
                          '-y',
                          '-r', '{}'.format(self.fps),  # choose fps
                          '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                          '-pix_fmt', 'argb',  # format
                          '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                          '-vcodec', 'mpeg4',
                          # '-b:v', '2000k',
                          '-qscale', '1',
                          self.filename)  # output encoding

        self.ffmpeg = subprocess.Popen(self.cmdstring, stdin = subprocess.PIPE, bufsize = -1)

    def send_frame_to_ffmpeg(self):
        self.fig.canvas.draw()
        string = self.fig.canvas.tostring_argb()

        self.ffmpeg.stdin.write(string)

    def cleanup(self):
        self.ffmpeg.communicate()


class SphericalSliceAnimator(CylindricalSliceAnimator):
    def __init__(self, simulation, cluster = False):
        super(SphericalSliceAnimator, self).__init__(simulation, cluster = cluster)

    def initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))
        self.fig.set_tight_layout(True)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [3, 1])
        self.ax_mesh = plt.subplot(grid_spec[0], projection = 'polar')
        self.ax_time = plt.subplot(grid_spec[1])

        self.ax_mesh.grid(True, color = 'pink', linewidth = 1, linestyle = ':')  # change grid color to make it show up against the colormesh
        self.ax_mesh.set_thetagrids(np.arange(0, 360, 30), frac = 1.05)
        self.ax_time.grid()

        self.ax_time.set_xlabel('Time (as)', fontsize = 18)
        self.ax_time.set_ylabel('Ionization Metric', fontsize = 18)

        self.ax_time.set_xlim(self.sim.times[0] / un.asec, self.sim.times[-1] / un.asec)
        self.ax_time.set_ylim(0, 1)

        self.ax_mesh.tick_params(axis = 'both', which = 'major', labelsize = 12)  # increase size of tick labels
        self.ax_mesh.tick_params(axis = 'y', which = 'major', colors = 'pink')  # make r ticks a color that shows up against the colormesh

        self.ax_mesh.set_rlabel_position(10)
        last_r_label = self.ax_mesh.get_yticklabels()[-1]
        last_r_label.set_color('black')  # last r tick is outside the colormesh, so make it black again

        self.ax_time.tick_params(labelright = True)
        self.ax_time.tick_params(axis = 'both', which = 'major', labelsize = 12)

        self.mesh, self.mesh_mirror = self.sim.mesh.attach_g_to_axis(self.ax_mesh, normalize = self.spec.animation_normalize, log = self.spec.animation_log_g)

        self.cbar_axis = self.fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
        self.cbar = plt.colorbar(mappable = self.mesh, cax = self.cbar_axis)
        self.cbar.ax.tick_params(labelsize = 12)

        if self.spec.animation_overlay_probability_current:
            self.quiver = self.sim.mesh.attach_probability_current_quiver(self.ax_mesh)

        if self.spec.electric_potential is not None:
            self.pulse_max = np.max(self.spec.electric_potential.get_amplitude(self.sim.times))
            self.electric_field_line, = self.ax_time.plot(self.sim.times / un.asec, np.abs(self.sim.electric_field_amplitude_vs_time / self.pulse_max),
                                                          label = r'Normalized $|E|$ Field', color = 'red', linewidth = 2)

        self.norm_line, = self.ax_time.plot(self.sim.times / un.asec, self.sim.norm_vs_time,
                                            label = 'Wavefunction Norm', color = 'black', linestyle = '--', linewidth = 3)

        self.overlaps_stackplot = self.ax_time.stackplot(self.sim.times / un.asec, *self.compute_stackplot_overlaps(),
                                                         labels = ['Initial State Overlap', r'Overlap with $n \leq 5$'], colors = ['.3', '.5'])

        self.time_line, = self.ax_time.plot([self.sim.times[self.sim.time_index] / un.asec, self.sim.times[self.sim.time_index] / un.asec], [0, 1],
                                            linestyle = '-.', color = 'gray')

        self.ax_mesh.axis('tight')
        self.ax_time.legend(loc = 'center left', fontsize = 12)

    def update_mesh_axis(self):
        self.sim.mesh.update_g_mesh(self.mesh, normalize = self.spec.animation_normalize, log = self.spec.animation_log_g)
        self.sim.mesh.update_g_mesh(self.mesh_mirror, normalize = self.spec.animation_normalize, log = self.spec.animation_log_g)

        if self.spec.animation_overlay_probability_current:
            self.sim.mesh.update_probability_current_quiver(self.quiver)
