API Reference
=============

.. currentmodule:: simulacra

Specifications and Simulations
------------------------------

The core of the Simulacra framework are the :class:`Specification` and :class:`Simulation` classes.
A :class:`Specification` collects the data required to run a simulation, but doesn't do any actual computation.
The specification can then be used to to generate a :class:`Simulation` via :func:`Specification.to_sim`, which will perform actual computations via the hook method :func:`Simulation.run`.

The :class:`Beet` is the superclass of both :class:`Specification` and :class:`Simulation`.
It provides a common interface for saving, loading, and cloning operations, as well as storing a unique identifier.

.. currentmodule:: simulacra.sims

.. autoclass:: simulacra.Beet

    .. automethod:: clone

.. autoclass:: simulacra.Specification

    .. automethod:: to_sim

    .. automethod:: save

    .. automethod:: load

    .. automethod:: info

.. autoclass:: simulacra.Simulation

    .. automethod:: run

    .. automethod:: save

    .. automethod:: load

.. autoclass:: Status

Info
----

Simulacra provides a system for hierarchically displaying information from nested objects.
To participate, an object should define an ``info()`` method that takes no arguments and returns an :class:`Info` instance which it gets from calling ``super().info()``.
Inside this method more fields and :class:`Info` objects can be added to the top-level :class:`Info`, which could represent information from attributes and nested objects, respectively.

.. currentmodule:: simulacra.info

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

Simulacra's high-level plotting functions are intended for quickly generating plots with a wide variety of basic graphical options.

.. autofunction:: xy_plot

.. autofunction:: xxyy_plot


.. autofunction:: xyz_plot


Low-Level Plotting Utilities
++++++++++++++++++++++++++++

The low-level plotting interface is designed to individually wrap common visualization tasks such as creating and saving ``matplotlib`` figures and setting axis options.

.. autoclass:: simulacra.vis.FigureManager

.. autofunction:: simulacra.vis.get_figure

.. autofunction:: simulacra.vis.save_current_figure

Animation Tools
+++++++++++++++

.. autofunction:: xyt_plot

.. autofunction:: xyzt_plot

.. autofunction:: animate

Simulation Animators
++++++++++++++++++++

:class:`Animator` and :class:`AxisManager` provide a method for a :class:`Simulation` to produce an animation while it's running.

.. autoclass:: AxisManager

.. autoclass:: Animator


Math
----

Simulacra's math library provides a few miscellaneous objects and functions with no particular focus.

.. currentmodule:: simulacra.math

.. autoclass:: SphericalHarmonic

.. autofunction:: complex_quad

.. autofunction:: complex_dblquad

.. autofunction:: complex_nquad

Summables
---------

:class:`Summand` and :class:`Sum` implement a composite pattern, where summands are summed to form sums, which delegate calls to the summands inside them.
For example, a :class:`Summand` could be the electric field of a single particle, and the electric field of a group of particles could be a sum of those electric fields.

.. currentmodule:: simulacra.summables

.. autoclass:: Summand

.. autoclass:: Sum

Units
-----

.. currentmodule:: simulacra.units

.. autofunction:: uround

.. autofunction:: get_unit_value_and_latex_from_unit

Utilities
---------

Simulacra's utility module provides a wide range of functions for working with arrays, simulations, log messages, attributes, and processes that doesn't quite fit anywhere else.

.. currentmodule:: simulacra.utils

.. autofunction:: memoize

.. autofunction:: multi_map

.. autofunction:: get_now_str

.. autofunction:: ensure_dir_exists

.. autofunction:: find_nearest_entry

.. autofunction:: find_or_init_sim

.. autoclass:: LogManager

.. autofunction:: timed

.. autoclass:: BlockTimer

.. autofunction:: get_file_size

.. autofunction:: get_file_size_as_string

.. autofunction:: try_loop

.. autoclass:: SubprocessManager


Cluster
-------

Interfacing with a Cluster
++++++++++++++++++++++++++

Simulacra provides an object-oriented data processing interface that can talk to a cluster, load simulations, and perform analysis on them.

.. currentmodule:: simulacra.cluster

.. autoclass:: ClusterInterface

   .. automethod:: cmd

   .. automethod:: remote_path_to_local_path

   .. automethod:: get_file

   .. automethod:: is_file_synced

   .. automethod:: mirror_file

   .. automethod:: walk_remote_path

   .. automethod:: mirror_dir

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

.. currentmodule:: simulacra.exceptions

.. autoexception:: SimulacraException
