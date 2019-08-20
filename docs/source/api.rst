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
    :members:

.. autoclass:: simulacra.Specification
    :members:

.. autoclass:: simulacra.Simulation
    :members:

.. autoclass:: Status
    :members:

Running Simulations
+++++++++++++++++++

.. autofunction:: find_sim_or_init
.. autofunction:: run_from_cache


Info
----

Simulacra provides a system for hierarchically displaying information from
nested objects.
To participate, an object should define an ``info()`` method that takes no
arguments and returns an :class:`Info` instance which it gets from calling
``super().info()``, or creates a new :class:`Info` if it is a root object.
Inside this method more fields and :class:`Info` objects can be added to the
top-level :class:`Info`, which could represent information from attributes and
nested objects, respectively.

.. currentmodule:: simulacra.info

.. autoclass:: Info

   .. automethod:: add_field

   .. automethod:: add_fields

   .. automethod:: add_info

   .. automethod:: add_infos


Math
----

.. currentmodule:: simulacra.math

Simulacra's math library provides a few miscellaneous objects and functions with no particular focus.

.. autofunction:: rand_phases
.. autofunction:: rand_phases_like

.. autoclass:: SphericalHarmonic

.. autofunction:: complex_quad
.. autofunction:: complex_quadrature
.. autofunction:: complex_dblquad
.. autofunction:: complex_nquad


Units
-----

.. currentmodule:: simulacra.units

.. autofunction:: get_unit_value
.. autofunction:: get_unit_values
.. autofunction:: get_unit_value_and_latex


Creating Specifications and Jobs Programmatically
-------------------------------------------------

.. currentmodule:: simulacra.parameters

.. autoclass:: Parameter

.. autofunction:: expand_parameters

.. autofunction:: ask_for_input
.. autofunction:: ask_for_bool
.. autofunction:: ask_for_choices
.. autofunction:: ask_for_eval

Utilities
---------

Simulacra's utility module provides a wide range of functions that don't quite
fit anywhere else.

.. currentmodule:: simulacra.utils

.. autofunction:: find_nearest_entry

.. autofunction:: ensure_parents_exist

.. autofunction:: get_file_size
.. autofunction:: bytes_to_str

.. autoclass:: LogManager
.. autoclass:: BlockTimer
.. autoclass:: SubprocessManager

.. autoclass:: StrEnum

Memoization
+++++++++++

.. autofunction:: memoize
.. autofunction:: cached_property
.. autofunction:: watched_memoize


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


Exceptions
----------

.. currentmodule:: simulacra

.. automodule:: simulacra.exceptions
   :members:
