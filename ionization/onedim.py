import datetime as dt
import functools
import logging
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nfft
import scipy as sp
import scipy.sparse as sparse
import scipy.special as special
from cycler import cycler

import compy as cp
import compy.cy as cy
from compy.units import *
from . import core, potentials

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


    # class SpectralMethod1DMesh:
    #     def __init__(self, parameters):
    #         self.parameters = parameters
    #
    #         self.x = np.linspace(-self.parameters.x_lim, self.parameters.x_lim, self.parameters.x_points)
    #         self.delta_x = np.abs(self.x[1] - self.x[0])
    #
    #         self.wavenumbers = 2 * pi * fft.fftshift(fft.fftfreq(len(self.x), d = self.delta_x))
    #
    #         self.psi = np.zeros(np.shape(self.x), dtype = np.complex128)
    #         self.potential = np.zeros(np.shape(self.x), dtype = np.complex128)
    #
    #     def fft(self):
    #         return fft.fft(self.psi, norm = 'ortho')
    #
    #     def ifft(self, ft):
    #         return fft.ifft(ft, norm = 'ortho')
    #
    #     def evolve_potential(self, delta_t):
    #         # print(np.exp(-1j * delta_t * self.potential))
    #         self.psi *= np.exp(-1j * delta_t * self.potential / hbar)
    #
    #     def evolve_free(self, delta_t):
    #         # print(np.exp(-1j * (self.wavenumbers ** 2) * delta_t))
    #         # print(np.exp(-1j * (hbar ** 2 / (2 * m_electron)) * (self.wavenumbers ** 2) * delta_t))
    #         # self.psi = self.ifft(fft.ifftshift(fft.fftshift(self.fft()) * np.exp(-1j * (self.wavenumbers ** 2) * delta_t)))
    #         # self.psi = self.ifft(fft.ifftshift(fft.fftshift(self.fft()) * np.exp(-1j * (hbar ** 2 / (2 * m_electron)) * (self.wavenumbers ** 2) * delta_t)))
    #         ft_shifted = fft.fftshift(self.fft())
    #         ft_shifted *= np.exp(-1j * (hbar / (2 * m_electron)) * (self.wavenumbers ** 2) * delta_t)
    #         ft_unshifted = fft.ifftshift(ft_shifted)
    #         self.psi = self.ifft(ft_unshifted)
    #
    #     def plot_potential(self, show = False, save = False, name = '', target_dir = None, img_format = 'png', title = None, **kwargs):
    #         plt.close()  # close any old figures
    #
    #         fig = plt.figure(figsize = (7, 7), dpi = 600)
    #         fig.set_tight_layout(True)
    #         axis = plt.subplot(111)
    #
    #         axis.plot(self.x / bohr_radius, self.potential / eV)
    #
    #         if save:
    #             save_current_figure(name = '{}_{}__potential'.format(self.parameters.name, name), target_dir = target_dir, img_format = img_format, **kwargs)
    #         if show:
    #             plt.show()
    #
    #         plt.close()
    #
    #     def plot_psi(self, show = False, save = False, name = '', target_dir = None, img_format = 'png', title = None, **kwargs):
    #         plt.close()  # close any old figures
    #
    #         fig = plt.figure(figsize = (7, 7), dpi = 600)
    #         fig.set_tight_layout(True)
    #         axis = plt.subplot(111)
    #
    #         axis.plot(self.x / bohr_radius, np.abs(self.psi) ** 2)
    #
    #         if save:
    #             save_current_figure(name = '{}_{}__psi'.format(self.parameters.name, name), target_dir = target_dir, img_format = img_format, **kwargs)
    #         if show:
    #             plt.show()
    #
    #         plt.close()
    #
    #     def plot_fft(self, show = False, save = False, name = '', target_dir = None, img_format = 'png', title = None, **kwargs):
    #         plt.close()  # close any old figures
    #
    #         fig = plt.figure(figsize = (7, 7), dpi = 600)
    #         fig.set_tight_layout(True)
    #         axis = plt.subplot(111)
    #
    #         axis.plot(self.wavenumbers, np.abs(fft.fftshift(self.fft())))
    #
    #         if save:
    #             save_current_figure(name = '{}_{}__fft'.format(self.parameters.name, name), target_dir = target_dir, img_format = img_format, **kwargs)
    #         if show:
    #             plt.show()
    #
    #         plt.close()


    # class SpectralMethodSimulation(Simulation):
    #     def __init__(self, parameters, **kwargs):
    #         super().__init__(parameters, **kwargs)
    #
    #         self.parameters = parameters
    #         self.mesh = SpectralMethod1DMesh(self.parameters)
    #
    #         total_time = self.parameters.time_final - self.parameters.time_initial
    #         self.times = np.linspace(self.parameters.time_initial, self.parameters.time_final, int(total_time / self.parameters.time_step) + 1)
    #         self.time_index = 0
    #         self.time_steps = len(self.times)
    #
    #     def run_simulation(self):
    #         self.mesh.evolve_free(self.parameters.time_step / 2)
    #
    #         while True:
    #             self.logger.debug('Working on time step {} / {} ({}%)'.format(self.time_index + 1, self.time_steps, np.around(100 * (self.time_index + 1) / self.time_steps, 2)))
    #
    #             self.mesh.evolve_potential(self.parameters.time_step)
    #
    #             self.time_index += 1
    #             if self.time_index == self.time_steps:
    #                 break
    #
    #             self.mesh.evolve_free(self.parameters.time_step)
    #
    #         self.mesh.evolve_free(self.parameters.time_step / 2)
    #
    #
    # class LineAnimator:
    #     def __init__(self, simulation, cluster = False):
    #         self.simulation = simulation
    #         self.parameters = simulation.parameters
    #         self.cluster = cluster
    #
    #         plt.set_cmap(plt.cm.viridis)
    #         plt.rcParams['xtick.major.pad'] = 5
    #         plt.rcParams['ytick.major.pad'] = 5
    #
    #         ideal_frame_count = self.parameters.animation_time * self.parameters.animation_fps
    #         self.animation_decimation = int(self.simulation.time_steps / ideal_frame_count)
    #         if self.animation_decimation < 1:
    #             self.animation_decimation = 1
    #         self.fps = (self.simulation.time_steps / self.animation_decimation) / self.parameters.animation_time
    #
    #         if cluster:
    #             self.filename = self.simulation.parameters.pickle_name + '.mp4'
    #         else:
    #             if self.parameters.animation_dir is not None:
    #                 target_dir = self.parameters.animation_dir
    #             else:
    #                 target_dir = os.getcwd()
    #
    #             postfix = ''
    #             if self.parameters.animation_log_g:
    #                 postfix += '_log'
    #
    #             self.filename = os.path.join(target_dir, '{}{}.mp4'.format(self.parameters.name, postfix))
    #
    #         ensure_dir_exists(self.filename)
    #
    #         try:
    #             os.remove(self.filename)
    #         except FileNotFoundError:
    #             pass
    #
    #         self.cmdstring = None
    #         self.ffmpeg = None
    #
    #     def initialize_figure(self):
    #         self.fig = plt.figure(figsize = (16, 12))
    #         self.fig.set_tight_layout(True)
    #
    #         # grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [3, 1])
    #         # self.ax_mesh = plt.subplot(grid_spec[0])
    #         # self.ax_time = plt.subplot(grid_spec[1])
    #         #
    #         # self.ax_mesh.grid()
    #         # self.ax_time.grid()
    #         #
    #         # self.ax_mesh.set_xlabel(r'$z$ (Bohr radii)', fontsize = 18)
    #         # self.ax_mesh.set_ylabel(r'$\rho$ (Bohr radii)', fontsize = 18)
    #         # self.ax_time.set_xlabel('Time (as)', fontsize = 18)
    #         # self.ax_time.set_ylabel('Ionization Metric', fontsize = 18)
    #         #
    #         # self.ax_time.set_xlim(self.simulation.times[0] / asec, self.simulation.times[-1] / asec)
    #         # self.ax_time.set_ylim(0, 1)
    #         #
    #         # self.ax_time.tick_params(labelright = True)
    #         # self.ax_time.tick_params(axis = 'both', which = 'major', labelsize = 12)
    #         # self.ax_mesh.tick_params(axis = 'both', which = 'major', labelsize = 12)
    #         #
    #         # self.mesh = self.simulation.mesh.attach_g_to_axis(self.ax_mesh, normalize = self.parameters.animation_normalize, log = self.parameters.animation_log_g, plot_limit = self.parameters.animation_plot_limit)
    #         #
    #         # divider = make_axes_locatable(self.ax_mesh)
    #         # cax = divider.append_axes("right", size = "2%", pad = 0.05)
    #         # self.cbar = plt.colorbar(cax = cax, mappable = self.mesh)
    #         # self.cbar.ax.tick_params(labelsize = 12)
    #         #
    #         # if self.parameters.animation_overlay_probability_current:
    #         #     self.quiver = self.simulation.mesh.attach_probability_current_quiver(self.ax_mesh, plot_limit = self.parameters.animation_plot_limit)
    #         #
    #         # self.pulse_max = self.parameters.external_potential.get_peak_amplitude()
    #         # self.external_potential_line, = self.ax_time.plot(self.simulation.times / asec, np.abs(self.simulation.external_potential_amplitude_vs_time / self.pulse_max),
    #         #                                                   label = r'Normalized $|E|$',
    #         #                                                   color = 'red', linewidth = 2)
    #         #
    #         # self.norm_line, = self.ax_time.plot(self.simulation.times / asec, self.simulation.norm_vs_time,
    #         #                                     label = r'$\left\langle \psi|\psi \right\rangle$',
    #         #                                     color = 'black', linestyle = '--', linewidth = 3)
    #         #
    #         # self.overlaps_stackplot = self.ax_time.stackplot(self.simulation.times / asec, *self.compute_stackplot_overlaps(),
    #         #                                                  labels = [r'$\left| \left\langle \psi|\psi_{init} \right\rangle \right|^2$', r'$\left| \left\langle \psi|\psi_{n\leq5} \right\rangle \right|^2$'],
    #         #                                                  colors = ['.3', '.5'])
    #         #
    #         # self.time_line, = self.ax_time.plot([self.simulation.times[self.simulation.time_index] / asec, self.simulation.times[self.simulation.time_index] / asec], [0, 1],
    #         #                                     linestyle = '-.', color = 'gray')
    #         #
    #         # self.ax_mesh.axis('tight')
    #
    #     def update_frame(self):
    #         self.update_mesh_axis()
    #         self.update_time_axis()
    #
    #     def update_mesh_axis(self):
    #         self.simulation.mesh.update_g_mesh(self.mesh, normalize = self.parameters.animation_normalize, log = self.parameters.animation_log_g, plot_limit = self.parameters.animation_plot_limit)
    #
    #         if self.parameters.animation_overlay_probability_current:
    #             self.simulation.mesh.update_probability_current_quiver(self.quiver, plot_limit = self.parameters.animation_plot_limit)
    #
    #     def update_time_axis(self):
    #         self.external_potential_line.set_ydata(np.abs(self.simulation.external_potential_amplitude_vs_time / self.pulse_max))
    #         self.norm_line.set_ydata(self.simulation.norm_vs_time)
    #         self.overlaps_stackplot = self.ax_time.stackplot(self.simulation.times / asec, *self.compute_stackplot_overlaps(),
    #                                                          labels = ['Initial State Overlap', r'Overlap with $n \leq 5$'], colors = ['.3', '.5'])
    #         self.time_line.set_xdata([self.simulation.times[self.simulation.time_index] / asec, self.simulation.times[self.simulation.time_index] / asec])
    #
    #     def initialize(self):
    #         self.initialize_figure()
    #
    #         self.ax_time.legend(loc = 'center left', fontsize = 14)  # legend must be created here so that it catches all of the lines in ax_time
    #
    #         canvas_width, canvas_height = self.fig.canvas.get_width_height()
    #
    #         self.cmdstring = ("ffmpeg",
    #                           '-y',
    #                           '-r', '{}'.format(self.fps),  # choose fps
    #                           '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
    #                           '-pix_fmt', 'argb',  # format
    #                           '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
    #                           '-vcodec', 'mpeg4',
    #                           # '-b:v', '2000k',
    #                           '-qscale', '1',
    #                           self.filename)  # output encoding
    #
    #         self.ffmpeg = subprocess.Popen(self.cmdstring, stdin = subprocess.PIPE, bufsize = -1)
    #
    #     def send_frame_to_ffmpeg(self):
    #         self.fig.canvas.draw()
    #         string = self.fig.canvas.tostring_argb()
    #
    #         self.ffmpeg.stdin.write(string)
    #
    #     def cleanup(self):
    #         self.ffmpeg.communicate()
    #
    #
    # if __name__ == '__main__':
    #     out_dir = out_dir = os.path.join(os.path.dirname(os.getcwd()), 'out', 'fft_test')
    #
    #     par = SpectralMethod1DParameters('spectral', x_points = 2 ** 14, x_lim = 100 * bohr_radius, time_final = 2500 * asec, time_step = 1 * asec)
    #     sim = SpectralMethodSimulation(par, log_console_level = logging.DEBUG)
    #
    #     # sim.mesh.psi = np.exp(-(sim.mesh.x ** 2) / ((.1 * bohr_radius) ** 2)) * np.exp(1j * (2 * pi / (500 * bohr_radius)) * sim.mesh.x)
    #     # sim.mesh.psi = np.ones(np.shape(sim.mesh.x))
    #     # sim.mesh.psi = np.abs(sim.mesh.x) * (sim.parameters.x_lim - np.abs(sim.mesh.x))
    #     # sim.mesh.potential[np.abs(sim.mesh.x) > 50 * bohr_radius] = 1000 * eV
    #
    #     delta = 2 * bohr_radius
    #     # # k = 2 * pi / 1 * bohr_radius
    #     sim.mesh.psi = np.exp(-0.5 * ((sim.mesh.x / delta) ** 2))
    #     sim.mesh.potential[np.abs(sim.mesh.x) > 80 * bohr_radius] = -1j * 1 * eV
    #
    #     sim.mesh.plot_fft(save = True, target_dir = out_dir)
    #     sim.mesh.plot_potential(save = True, target_dir = out_dir)
    #     sim.mesh.plot_psi(save = True, target_dir = out_dir, name = 'before')
    #
    #     sim.run_simulation()

    # print(len(sim.mesh.x))
    # print(sim.mesh.delta_x / bohr_radius)
    # print(sim.mesh.wavenumbers)

    # sim.mesh.plot_psi(save = True, target_dir = out_dir, name = 'after')




    # lim = 1000
    # n = 2 ** 14
    # x = np.linspace(-lim, lim, n)
    # delta_x = x[1] - x[0]
    #
    # print(n)
    # print(x)
    # print(delta_x)
    #
    # w_cut = 5
    # k = 3
    # # func = sinc(w_cut * t)
    # # func = np.exp(1j * 2 * np.pi * f_cut * t)
    # # func = np.sin(k * t)
    # func = np.exp((-x ** 2 / (5 ** 2)) + (1j * 0 * x))
    #
    # delta_t = 50
    #
    # plt.plot(x, np.abs(func) ** 2)
    # plt.show()
    #
    # fft_func, fft_freq = angular_fft(func, x)
    #
    # plt.plot(fft.fftshift(fft_freq), fft.fftshift(np.abs(fft_func) ** 2))
    # plt.show()
    #
    # freq, f = fft.fftshift(fft_freq), fft.fftshift(fft_func)
    #
    # fft_func = fft.ifftshift(f * np.exp(-1j * (freq ** 2) * delta_t))
    #
    # print(np.exp(-1j * (freq ** 2) * delta_t))
    # print((freq ** 2) * delta_t)
    #
    # plt.plot(x, np.abs(angular_ifft(fft_func)) ** 2)
    # plt.show()

#
# # fft_func = fft.fft(func, norm = 'ortho')
# # fft_freq = fft.fftfreq(n, d = delta_t)
# #
# # print(fft_freq)
# #
# # # plt.axvline(1 / (2 * np.pi))
# #
# # plt.plot(fft.fftshift(fft_freq), fft.fftshift(np.abs(fft_func)))
# # plt.show()
# #
# # ifft_func = fft.ifft(fft_func, norm = 'ortho')
# #
# # plt.plot(t, ifft_func)
# # plt.show()
