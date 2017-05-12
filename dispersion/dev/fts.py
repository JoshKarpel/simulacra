import os

import numpy as np
import scipy.interpolate as interp
import scipy.signal as psignal

import compy as cp
import plots
from units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
# OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME + '_unmodulated')
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

plt_kwargs = dict(
    target_dir = OUT_DIR
)

if __name__ == '__main__':
    time, pd_signal, piezo = np.loadtxt(os.path.join(os.getcwd(), 'fts_data.txt'), delimiter = ',', unpack = True)

    dt = np.abs(time[1] - time[0])
    print('dt', dt)

    pd_signal_max_indices = psignal.argrelmax(pd_signal, order = 4, mode = 'clip')[0]
    pd_signal_max_index_diffs = np.diff(pd_signal_max_indices)
    print(pd_signal_max_indices)
    print(pd_signal_max_index_diffs)

    plots.xy_plot('photodiode_signal_vs_time',
                  time,
                  pd_signal,
                  x_label = 'Time', y_label = 'Photodiode Signal',
                  vlines = [time[ii] for ii in pd_signal_max_indices],
                  **plt_kwargs
                  )

    plots.xy_plot('photodiode_signal_vs_time__zoom',
                  time,
                  pd_signal,
                  x_label = 'Time', y_label = 'Photodiode Signal',
                  vlines = [time[ii] for ii in pd_signal_max_indices],
                  x_lower_limit = .4, x_upper_limit = .425,
                  **plt_kwargs
                  )

    plots.xy_plot('piezo_voltage_vs_time',
                  time,
                  piezo,
                  x_label = 'Time', y_label = 'Piezo Voltage',
                  **plt_kwargs
                  )

    avg_of_index_diff = np.average(pd_signal_max_index_diffs)
    std_dev_of_index_diff = np.std(pd_signal_max_index_diffs)

    print(avg_of_index_diff, std_dev_of_index_diff)

    cleaner = np.abs(pd_signal_max_index_diffs - avg_of_index_diff) < std_dev_of_index_diff

    cleaned_indices = pd_signal_max_indices[cleaner]
    cleaned_diffs = pd_signal_max_index_diffs[cleaner]

    print(cleaned_diffs)
    print(len(pd_signal_max_index_diffs), len(cleaned_diffs))
