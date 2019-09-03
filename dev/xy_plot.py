import logging

from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = Path(__file__).stem
OUT_DIR = Path(__file__).parent / "out" / FILE_NAME

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
        figman = si.vis.xy_plot(
            "test",
            x,
            *y,
            line_kwargs=({"linestyle": "-"}, {"linestyle": ":", "color": "teal"}, None),
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
            target_dir=OUT_DIR,
        )
