import logging
import os

import numpy as np

import compy as cp

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', stdout_logs = True, stdout_level = logging.DEBUG, file_dir = OUT_DIR, file_logs = False) as logger:
        pi = 3.1415
        twopi = 2 * pi

        x = np.linspace(0, 10, 1000)
        y = [np.sin(x), np.cos(x), np.arctan(x)]

        plt_kwargs = dict(
            line_kwargs = ({'linestyle': '-'}, {'linestyle': ':', 'color': 'teal'}),  # note that we don't need to explicitly add None for the third line
            line_labels = (r'$\sin(x)$', r'$\cos(x)$', r'$\arctan(x)$'),
            x_scale = 1, y_scale = 'mm',
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
        ]

        for extras in extra_kwargs:
            kwargs = {**plt_kwargs, **extras}

            path = cp.utils.xy_plot('test', x, *y,
                                    **kwargs)

            print(path)
