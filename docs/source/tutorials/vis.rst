Basic Visualization
===================

.. currentmodule:: simulacra.vis

Simulacra provides a high-level interface for quickly producing line and heatmap
plots. They are generally not going to be of production-level quality, but they
provide powerful shortcuts for exploratory work.

Let's start by making a simple line plot.

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
