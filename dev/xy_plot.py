import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra",
        stdout_logs=True,
        stdout_level=logging.DEBUG,
        file_dir=OUT_DIR,
        file_logs=False,
    ) as logger:
        x = np.linspace(0, 10, 1000)
        y = [np.sin(x), np.cos(x), np.arctan(x)]

        plt_kwargs = dict(
            line_kwargs=(
                {"linestyle": "-"},
                {"linestyle": ":", "color": "teal"},
                None,
            ),  # note that we don't need to explicitly add None for the third line
            line_labels=(r"$\sin(x)$", r"$\cos(x)$", r"$\arctan(x)$"),
            x_unit=1,
            y_unit="mm",
            hlines=(-0.5, 0.2, 0.33),
            hline_kwargs=({"color": "blue"}, {"color": "orange"}, None),
            vlines=(2, 4, u.twopi),
            vline_kwargs=(None, {"color": "red", "linestyle": "-."}, None),
            x_extra_ticks=(u.pi, u.pi / 2),
            x_extra_tick_labels=(r"$\pi$", r"$\frac{\pi}{2}$"),
            y_extra_ticks=(0.66, 0.88),
            y_extra_tick_labels=(r"$\alpha$", r"$\beta$"),
            title="foo",
            x_label="bar",
            y_label="$baz$",
            save_csv=True,
            img_format="png",
            target_dir=OUT_DIR,
        )

        extra_kwargs = [
            dict(name_postfix=""),
            # dict(name_postfix = 'scale=1', fig_scale = 1),
            # dict(name_postfix = 'scale=1_tight', fig_scale = 1),
            # dict(name_postfix = 'scale=1_dpi=2', fig_scale = 1, fig_dpi_scale = 2),
            # dict(name_postfix = 'scale=2', fig_scale = 2),
            # dict(name_postfix = 'scale=2_dpi=2', fig_scale = 2, fig_dpi_scale = 2),
            # dict(name_postfix = 'scale=1_dpi=6', fig_scale = 1, fig_dpi_scale = 6),
            # dict(name_postfix = 'logX', x_log_axis = True),
            # dict(name_postfix = 'logY', y_log_axis = True),
            # dict(name_postfix = 'logXY', x_log_axis = True, y_log_axis = True),
            # dict(name_postfix = 'square', equal_aspect = True, fig_dpi_scale = 6,),
        ]

        for extras in extra_kwargs:
            figman = si.vis.xy_plot(
                "test" + extras.pop("name_postfix"), x, *y, **plt_kwargs, **extras
            )

            print(figman)
