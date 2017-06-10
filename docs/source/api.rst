API Reference
=============

Specifications and Simulations
------------------------------

.. currentmodule:: simulacra

.. autoclass:: Specification

   .. automethod:: to_simulation

   .. automethod:: clone

   .. automethod:: save

   .. automethod:: load

   .. automethod:: info

.. autoclass:: Simulation

   .. automethod:: run_simulation

   .. automethod:: save

   .. automethod:: load

   .. automethod:: info


Info
----

.. autoclass:: Info

   .. automethod:: add_field

   .. automethod:: add_fields

   .. automethod:: add_info

   .. automethod:: add_infos

Visualization
-------------

.. currentmodule:: simulacra.vis

High-Level Plotting Functions
+++++++++++++++++++++++++++++


.. autofunction:: xy_plot

.. autofunction:: xxyy_plot

.. autofunction:: xyt_plot

.. autofunction:: xyz_plot

.. autofunction:: xyzt_plot

Low-Level Plotting Utilities
++++++++++++++++++++++++++++

.. autoclass:: simulacra.vis.FigureManager

.. autofunction:: simulacra.vis.get_figure

.. autofunction:: simulacra.vis.save_current_figure

Math
----

.. currentmodule:: simulacra.math

.. autoclass:: SphericalHarmonic

.. autofunction:: complex_quad

.. autofunction:: complex_dblquad

.. autofunction:: complex_nquad

Summables
---------

.. currentmodule:: simulacra

.. autoclass:: Summand

.. autoclass:: Sum

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

.. autofunction:: find_or_init_sim

Cluster
-------

Interfacing with a Cluster
++++++++++++++++++++++++++

.. currentmodule:: simulacra.cluster

.. autoclass:: ClusterInterface

   .. automethod:: cmd

   .. automethod:: remote_path_to_local_path

   .. automethod:: get_file

   .. automethod:: put_file

   .. automethod:: is_file_synced

   .. automethod:: mirror_file

   .. automethod:: walk_remote_path

   .. automethod:: mirror_remote_home_dir

.. autoclass:: SimulationResult

.. autoclass:: JobProcessor

   .. automethod:: save

   .. automethod:: load

   .. automethod:: load_sims

   .. automethod:: summarize

   .. automethod:: select_by_kwargs

   .. automethod:: select_by_lambda

   .. automethod:: parameter_set

Creating Specifications and Jobs Programmatically
+++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: Parameter

.. autofunction:: expand_parameters_to_dicts

Exceptions
----------

.. currentmodule:: simulacra

.. autoexception:: SimulacraException
