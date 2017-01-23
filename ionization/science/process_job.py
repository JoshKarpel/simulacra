import sys
import os
import logging
import itertools as it
from pprint import pprint

sys.path.append(r"D:\GitHubProjects\compy")
sys.path.append(r"C:\Users\Josh\GitHubProjects\compy")

import compy as cp
from compy.units import *
import ionization as ion
import ionization.cluster as clu

import matplotlib as mpl

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        job_dir = os.getcwd()
        job_name = os.path.split(job_dir)[-1]

        OUT_DIR = os.path.join(job_dir, 'plots')

        jp = clu.SincPulseJobProcessor.load(os.path.join(job_dir, job_name + '.job'))

        jp.make_plot('diag',
                     'file_name',
                     clu.KeyFilterLine('run_time', label = 'Run Time'),
                     clu.KeyFilterLine('elapsed_time', label = 'Elapsed Time'),
                     target_dir = OUT_DIR,
                     y_scale = 'hours', y_label = 'Time to Complete',
                     x_label = 'Simulation Number', )

        logger.info('Made diagnostics plot')

        phases, fluences = tuple(sorted(jp.parameter_sets['phase'])), tuple(sorted(jp.parameter_sets['fluence']))

        # PLOTS OF METRICS AGAINST PULSE WIDTH, SKIPPING TO GET A BROAD OVERVIEW #

        colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d']

        fluences_skipping = list(fluences[i] for i in range(5))
        color_dict = dict(zip(fluences_skipping, colors))

        dashes = ((), (5, 2))
        dashes_dict = dict(zip(phases, dashes))

        for metric in ('final_initial_state_overlap', 'final_norm'):
            lines = [clu.KeyFilterLine(key = metric,
                                       filters = (clu.check_value_by_key('phase', phase), clu.check_value_by_key('fluence', fluence)),
                                       label = r'$\phi$ = {}, $F$ = {} $\mathrm{{J/cm^2}}$'.format(phase, uround(fluence, J / (cm ** 2), 3)),
                                       color = color_dict[fluence],
                                       dashes = dashes_dict[phase],
                                       linewidth = 2,
                                       )
                     for fluence in fluences_skipping for phase in sorted(phases)]

            y_label = ' '.join(t.title() for t in metric.split('_'))

            jp.make_plot(metric + '_vs_pulse_width',
                         'pulse_width',
                         *lines,
                         target_dir = OUT_DIR,
                         y_label = y_label,
                         y_lower_limit = 0, y_upper_limit = 1,
                         x_scale = 'asec', x_label = 'Pulse Width',
                         legend_on_right = True, )
            jp.make_plot(metric + '_vs_pulse_width_log',
                         'pulse_width',
                         *lines,
                         target_dir = OUT_DIR,
                         y_label = y_label, y_log_axis = True,
                         x_scale = 'asec', x_label = 'Pulse Width',
                         legend_on_right = True)

            logger.info('Made overview plot for metric {}'.format(metric))

        # PLOTS OF METRICS AGAINST PULSE WIDTH, FIVE AT A TIME #

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        dashes = ((), (5, 2))
        dashes_dict = dict(zip(phases, dashes))

        for metric in ('final_initial_state_overlap', 'final_norm'):
            for ii, fluence_group in enumerate(cp.utils.grouper(sorted(fluences), 5)):
                color_dict = dict(zip(sorted(fluence_group), colors))

                lines = [clu.KeyFilterLine(key = metric,
                                           filters = (clu.check_value_by_key('phase', phase), clu.check_value_by_key('fluence', fluence)),
                                           label = r'$\phi$ = {}, $F$ = {} $\mathrm{{J/cm^2}}$'.format(phase, uround(fluence, J / (cm ** 2), 3)),
                                           color = color_dict[fluence],
                                           dashes = dashes_dict[phase],
                                           linewidth = 2,
                                           )
                         for fluence in fluence_group for phase in sorted(phases)]

                out_dir_temp = os.path.join(OUT_DIR, metric)
                y_label = ' '.join(t.title() for t in metric.split('_'))

                jp.make_plot(metric + '_vs_pulse_width__{}'.format(ii),
                             'pulse_width',
                             *lines,
                             target_dir = out_dir_temp,
                             y_label = y_label,
                             y_lower_limit = 0, y_upper_limit = 1,
                             x_scale = 'asec', x_label = 'Pulse Width',
                             legend_on_right = True, )
                jp.make_plot(metric + '_vs_pulse_width_log__{}'.format(ii),
                             'pulse_width',
                             *lines,
                             target_dir = out_dir_temp,
                             y_label = y_label, y_log_axis = True,
                             x_scale = 'asec', x_label = 'Pulse Width',
                             legend_on_right = True)

                logger.info('Made grouped plot for metric {}, fluence group {}'.format(metric, ii))
