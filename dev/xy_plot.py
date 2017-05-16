import logging
import os

import numpy as np

import simulacra as si


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        pi = 3.1415
        twopi = 2 * pi

        x = np.linspace(0, 10, 1000)
        y = [np.sin(x), np.cos(x), np.arctan(x)]

        plt_kwargs = dict(
                line_kwargs = ({'linestyle': '-'}, {'linestyle': ':', 'color': 'teal'}),  # note that we don't need to explicitly add None for the third line
                line_labels = (r'$\sin(x)$', r'$\cos(x)$', r'$\arctan(x)$'),
                x_unit = 1, y_unit = 'mm',
                hlines = (-.5, .2, .33), hline_kwargs = ({'color': 'blue'}, {'color': 'orange'}, None),
                vlines = (2, 4, twopi), vline_kwargs = (None, {'color': 'red', 'linestyle': '-.'}, None),
                x_extra_ticks = (pi, pi / 2), x_extra_tick_labels = (r'$\pi$', r'$\frac{\pi}{2}$'),
                y_extra_ticks = (.66, .88), y_extra_tick_labels = (r'$\alpha$', r'$\beta$'),
                title = 'foo', x_label = 'bar', y_label = '$baz$',
                font_size_title = 22,
                save_csv = True,
                target_dir = OUT_DIR,
        )

        extra_kwargs = [
            dict(),
            dict(name_postfix = 'scale=1', img_format = 'png', img_scale = 1),
            dict(name_postfix = 'scale=2', img_format = 'png', img_scale = 2),
            dict(name_postfix = 'logX', x_log_axis = True),
            dict(name_postfix = 'logY', y_log_axis = True),
            dict(name_postfix = 'logXY', x_log_axis = True, y_log_axis = True),
        ]

        for extras in extra_kwargs:
            kwargs = {**plt_kwargs, **extras}

            path = si.plots.xy_plot('test', x, *y,
                                    **kwargs)

            print(path)
