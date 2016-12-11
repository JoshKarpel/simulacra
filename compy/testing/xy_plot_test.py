import logging
import os

import numpy as np

import compy as cp

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        x = np.linspace(0, 10, 1000)
        y = [np.sin(x), np.cos(x)]

        path, axis = cp.utils.xy_plot('test', x, *y,
                                      line_kwargs = ({'linestyle': '--'}, {'linestyle': ':', 'color': 'teal'}),
                                      line_labels = ('sin', 'cos'),
                                      x_scale = 2, y_scale = 'mm',
                                      hlines = (.1, .2, .33), hline_kwargs = ({'color': 'blue'}, {'color': 'orange'},),
                                      vlines = (2, 4, 7.5), vline_kwargs = (None, {'color': 'red', 'linestyle': '-.'},),
                                      title = 'foo', x_label = 'bar', y_label = '$baz$',
                                      font_size_title = 22,
                                      save_csv = True,
                                      target_dir = OUT_DIR)

        print(path)
        print(axis)
