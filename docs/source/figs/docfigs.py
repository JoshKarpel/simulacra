import numpy as np
import simulacra.vis as vis

for img_format in ('png', 'pdf'):
    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.exp(np.sin(x))

    vis.xy_plot(
        'y_vs_x',
        x, y,
        x_label = r'$x$', y_label = r'$ e^{\sin(x)} $',
        img_format = img_format,
    )

    vis.xy_plot(
        'y_vs_x__v2',
        x, y,
        x_label = r'$x$', x_unit = 'rad',
        y_label = r'$ e^{\sin(x)} $',
        img_format = img_format,
    )
