import os
import logging

import numpy as np

import matplotlib

matplotlib.use('pgf')

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'figs__' + FILE_NAME)

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
    "axes.labelsize": 11,  # LaTeX default is 10pt font.
    "font.size": 11,
    "legend.fontsize": 10,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
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
    plt.clf()
    fig = plt.figure(figsize = figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax


def save_figure(filename):
    cp.utils.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pdf')
    cp.utils.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pgf')


grid_kwargs = {
    'linestyle': ':',
    'color': 'black',
    'linewidth': .5,
    'alpha': 0.5
}


def make_fig_sinc_pulse_power_spectrum():
    fig, ax = get_figure()

    lower = .15
    upper = .85
    carrier = (lower + upper) / 2

    omega = np.linspace(-1, 1, 1000)
    power = np.where(np.abs(omega) < upper, 1, 0) * np.where(np.abs(omega) > lower, 1, 0)

    ax.fill_between(omega, 0, power, alpha = 1, edgecolor = 'black', facecolor = 'darkgray')

    plt.annotate(s = '', xy = (lower, .75), xytext = (upper, .75), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(carrier + .1 * (upper - carrier), .775, r'$\Delta$')

    plt.annotate(s = '', xy = (-lower, .75), xytext = (-upper, .75), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(-carrier - .1 * (lower - carrier), .775, r'$\Delta$')

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel(r'$   \omega  $')
    ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    ax.yaxis.set_label_coords(-.025, .5)

    ax.set_xticks([-upper, -carrier, -lower, 0, lower, carrier, upper])
    ax.set_xticklabels([r'$-\omega_{\mathrm{max}}$',
                        r'$-\omega_{\mathrm{carrier}}$',
                        r'$-\omega_{\mathrm{min}}$',
                        r'$0$',
                        r'$\omega_{\mathrm{min}}$',
                        r'$\omega_{\mathrm{carrier}}$',
                        r'$\omega_{\mathrm{max}}$'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r'$0$', r'$\left|   \mathcal{E}_{\omega}      \right|^2$'])

    ax.grid(True, **grid_kwargs)

    save_figure('sinc_pulse_power_spectrum')


if __name__ == '__main__':
    make_fig_sinc_pulse_power_spectrum()
