import sys
import os
import shutil
import logging
import argparse
import json
import pickle
import functools as ft

import numpy as np

import compy as cp
import compy.cluster as clu
import ionization as ion
import ionization.cluster as iclu
from compy.units import *


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization') as logger:
        jp_lide = cp.cluster.JobProcessor.load('compare_to_velocity.job')
        jp_vide = cp.cluster.JobProcessor.load('vide_compare.job')

        results_lide = list(jp_lide.data.values())
        results_vide = list(jp_vide.data.values())

        plt_kwargs = dict(
            x_unit = 'rad',
            target_dir = OUT_DIR,
        )

        for log in (True, False):
            postfix = ''
            if log:
                postfix += '__log'

            cp.plots.xy_plot(
                f'ionization_vs_phase__length' + postfix,
                [r.phase for r in results_lide],
                [r.final_bound_state_overlap for r in results_lide],
                y_log_axis = log,
                **plt_kwargs,
            )

            cp.plots.xy_plot(
                f'ionization_vs_phase__velocity' + postfix,
                [r.phase for r in results_vide],
                [r.final_bound_state_overlap for r in results_vide],
                y_log_axis = log,
                **plt_kwargs,
            )

            cp.plots.xy_plot(
                f'ionization_vs_phase__compare' + postfix,
                [r.phase for r in results_vide],
                [r.final_bound_state_overlap for r in results_lide],
                [r.final_bound_state_overlap for r in results_vide],
                line_labels = ('Length', 'Velocity'),
                y_log_axis = log,
                **plt_kwargs,
            )

            rel_lide = results_lide[0].final_bound_state_overlap
            rel_vide = results_vide[0].final_bound_state_overlap

            cp.plots.xy_plot(
                f'ionization_vs_phase__compare_rel' + postfix,
                [r.phase for r in results_vide],
                [r.final_bound_state_overlap / rel_lide for r in results_lide],
                [r.final_bound_state_overlap / rel_vide for r in results_vide],
                line_labels = ('Length (Rel.)', 'Velocity (Rel.)'),
                y_log_axis = log,
                **plt_kwargs,
            )
