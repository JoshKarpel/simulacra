import os
import sys
import logging

from tqdm import tqdm
import numpy as np

import matplotlib

matplotlib.use('pgf')

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = cp.utils.Logger('compy', 'ionization', stdout_level = logging.INFO)


def figsize(scale):
    fig_width_pt = 498.66258  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 9,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


# I make my own newfig and savefig functions
def get_figure(width = 0.9):
    if width == 'full':
        width = 0.9
    elif width == 'half':
        width = .45

    plt.clf()
    fig = plt.figure(figsize = figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax


def save_figure(filename):
    cp.utils.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pdf')
    cp.utils.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pgf')


def get_func_name():
    return sys._getframe(1).f_code.co_name


grid_kwargs = {
    'dashes': [.5, .5],
    'color': 'black',
    'linewidth': .5,
    'alpha': 0.4
}


def sinc_pulse_power_spectrum_full():
    fig, ax = get_figure('full')

    lower = .15
    upper = .85
    c = (lower + upper) / 2

    omega = np.linspace(-1, 1, 1000)
    power = np.where(np.abs(omega) < upper, 1, 0) * np.where(np.abs(omega) > lower, 1, 0)

    ax.fill_between(omega, 0, power, alpha = 1, edgecolor = 'black', facecolor = 'darkgray')

    delta_line_y_coord = .75
    plt.annotate(s = '', xy = (lower, delta_line_y_coord), xytext = (upper, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(c + .1 * (upper - c), delta_line_y_coord + .025, r'$\Delta$')

    plt.annotate(s = '', xy = (-lower, delta_line_y_coord), xytext = (-upper, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(-c - .1 * (lower - c), delta_line_y_coord + .025, r'$\Delta$')

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel(r'$   \omega  $')
    ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    ax.yaxis.set_label_coords(-.1, .2)

    ax.set_xticks([-upper, -c, -lower, 0, lower, c, upper])
    ax.set_xticklabels([r'$-\omega_{\mathrm{max}}$',
                        r'$-\omega_{\mathrm{c}}$',
                        r'$-\omega_{\mathrm{min}}$',
                        r'$0$',
                        r'$\omega_{\mathrm{min}}$',
                        r'$\omega_{\mathrm{c}}$',
                        r'$\omega_{\mathrm{max}}$'
                        ])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([
        r'$0$',
        r'$\left|   \mathcal{E}_{\omega}      \right|^2$',
    ])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


def sinc_pulse_power_spectrum_half():
    fig, ax = get_figure('half')

    lower = .15
    upper = .85
    carrier = (lower + upper) / 2

    omega = np.linspace(0, 1, 1000)
    power = np.where(np.abs(omega) < upper, 1, 0) * np.where(np.abs(omega) > lower, 1, 0)

    ax.fill_between(omega, 0, power, alpha = 1, edgecolor = 'black', facecolor = 'darkgray')

    delta_line_y_coord = .75
    plt.annotate(s = '', xy = (lower, delta_line_y_coord), xytext = (upper, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(carrier + .1 * (upper - carrier), delta_line_y_coord + 0.25, r'$\Delta$')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel(r'$   \omega  $')
    ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    ax.yaxis.set_label_coords(-.1, .2)

    ax.set_xticks([0, lower, carrier, upper])
    ax.set_xticklabels([r'$0$',
                        r'$\omega_{\mathrm{min}}$',
                        r'$\omega_{\mathrm{c}}$',
                        r'$\omega_{\mathrm{max}}$'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r'$0$', r'$\left|   \mathcal{E}_{\omega}      \right|^2$'])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


def sinc_pulse_electric_field():
    fig, ax = get_figure('half')

    time = np.linspace(-1, 1, 1000)
    field = cp.math.sinc(time) * np.cos(time)

    ax.plot(time, field, color = 'black', linewidth = 2)

    ax.set_xlim(-1, 1)
    # ax.set_ylim(-1.2, 1.2)

    # ax.set_xlabel(r'$   \omega  $')
    # ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    # ax.yaxis.set_label_coords(-.025, .5)
    #
    # ax.set_xticks([0, lower, c, upper])
    # ax.set_xticklabels([r'$0$',
    #                     r'$\omega_{\mathrm{min}}$',
    #                     r'$\omega_{\mathrm{c}}$',
    #                     r'$\omega_{\mathrm{max}}$'])
    # ax.set_yticks([0, 1])
    # ax.set_yticklabels([r'$0$', r'$\left|   \mathcal{E}_{\omega}      \right|^2$'])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


def gaussian_pulse_power_spectrum_half():
    # TODO: er, be caereful, is delta for the power spectrum or the amplitude spectrum?
    fig, ax = get_figure('half')

    carrier = .6
    sigma = .1
    delta = 2 * np.sqrt(2 * np.log(2)) * sigma

    omega = np.linspace(0, 1, 1000)
    power = np.exp(-.5 * (((omega - carrier) / sigma) ** 2)) / np.sqrt(twopi) / sigma
    max_power = np.max(power)
    power /= max_power

    ax.fill_between(omega, 0, power, alpha = 1, edgecolor = 'black', facecolor = 'darkgray')

    delta_line_y_coord = .3
    plt.annotate(s = '', xy = (carrier - delta / 2, delta_line_y_coord), xytext = (carrier + delta / 2, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(carrier + sigma / 5, delta_line_y_coord - .1, r'$\Delta$')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel(r'$   \omega  $')
    ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    ax.yaxis.set_label_coords(-.1, .2)

    ax.set_xticks([0, carrier, carrier - delta / 2, carrier + delta / 2])
    ax.set_xticklabels([r'$0$',
                        r'$  \omega_{\mathrm{c}}  $',
                        r'$  \omega_{\mathrm{c}} - \frac{\Delta}{2}   $',
                        r'$  \omega_{\mathrm{c}} + \frac{\Delta}{2}   $',
                        ])
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels([
        r'$0$',
        r'$\frac{1}{2}   \left|   \mathcal{E}_{\omega}    \right|^2$',
        r'$\left|   \mathcal{E}_{\omega}      \right|^2$',
    ])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


if __name__ == '__main__':
    with log as logger:
        figures = [
            # sinc_pulse_power_spectrum_full,
            # sinc_pulse_power_spectrum_half,
            # sinc_pulse_electric_field,
            gaussian_pulse_power_spectrum_half,
        ]

        for fig in tqdm(figures):
            fig()
