API Reference
=============

Specifications and Simulations
------------------------------

.. currentmodule:: simulacra

.. autoclass:: Specification

   .. automethod:: to_simulation

   .. automethod:: info

   .. automethod:: clone

   .. automethod:: save

   .. automethod:: load

.. autoclass:: Simulation

   .. automethod:: info

   .. automethod:: run_simulation

   .. automethod:: save

   .. automethod:: load

Visualization
-------------

High-Level Plotting Functions
+++++++++++++++++++++++++++++

.. currentmodule:: simulacra.plots

.. autofunction:: xy_plot

.. autofunction:: xyt_plot

.. autofunction:: xyz_plot

.. autofunction:: xyzt_plot

Low-Level Plotting Utilities
++++++++++++++++++++++++++++

.. autoclass:: FigureManager

.. autofunction:: get_figure

.. autofunction:: save_current_figure

Units
-----

.. currentmodule:: simulacra.units

.. autofunction:: uround

.. autofunction:: get_unit_value_and_latex_from_unit

Utilities
---------

.. currentmodule:: simulacra.utils

.. autofunction:: memoize

.. autofunction:: multi_map

.. autofunction:: ensure_dir_exists

.. autofunction:: find_nearest_entry

Cluster
-------

Interfacing with a Cluster
++++++++++++++++++++++++++

.. currentmodule:: simulacra.cluster

.. autoclass:: ClusterInterface

.. autoclass:: JobProcessor

Creating Jobs
+++++++++++++

.. autoclass:: Parameter

.. autofunction:: expand_parameters_to_dicts
