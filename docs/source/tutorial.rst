Tutorial
========

.. currentmodule:: simulacra

Visualization Tools
-------------------

The simplest and least-integrated way to use Simulacra is to use its visualization tools.
The tools come in two parts:

- A high-level interface for quickly producing line and heatmap plots with many features, either static or animated.
- A low-level interface for generating correctly-sized figures, setting axis options, etc.

Let's work with the high-level interface first and make a simple line plot.

::

    import numpy as np
    import simulacra.vis as vis

    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.exp(np.sin(x))

    vis.xy_plot('y_vs_x',
                x, y,
                x_label = r'$x$', y_label = r'$ \sin(x) \, \exp(x) $')

That code will produce a file ``y_vs_x.pdf`` in the current working directory, containing the image shown below:

.. image:: figs/y_vs_x.*
   :align: center

We can probably improve this a little.
``x`` is clearly measured in radians, so let's tell Simulacra that:

::

    vis.xy_plot('y_vs_x',
                x, y,
                x_label = r'$x$', x_unit = 'rad',
                y_label = r'$ \sin(x) \, \exp(x) $')

.. image:: figs/y_vs_x__v2.*
   :align: center

Simulations
-----------
