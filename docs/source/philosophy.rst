Simulacra's Philosophy
======================

Simulacra's implied architecture is that a :class:`Simulation` is created from a
:class:`Specification`, with no other possible configuration (because the
:class:`Simulation` constructor does not take any other arguments). In fact,
you should be creating simulations via :meth:`Specification.to_sim`, so you
don't even see the :class:`Simulation` constructor in the first place.
Why does Simulacra impose this separation?


Portability
-----------

In modern computing, it is often necessary to run code on remote resources.
Simulacra's design was motivated by high-throughput computing, where the vast
majority of the computation will be executed on wide-spread, inhomogeneous
resources.

A :class:`Specification` is a simple way to move all of the input parameters of
a calculation to a remote location. You simply save the specification to a file
using :meth:`Beet.save` (:class:`Beet` is the superclass of both
:class:`Specification` and :class:`Simulation`), move the file to the remote
machine, load it up, turn it into a simulation using
:meth:`Specification.to_sim` and run it.

Because the specification knows what kind of simulation it produces, you don't
even have to write specialized code to construct the simulation. That means
that the remote-execution code is completely generic over whatever concrete
specifications and simulations you have written. For example,
:func:`find_sim_or_init` will work with any :class:`Specification` you write.


Reproducibility
---------------

When we talk about reproducibility in science, we generally mean that if ``A``
performs an experiment and gets some result, ``B`` should be able to perform the
same experiment from scratch and get the same result.
In software, however, we often have trouble achieving an even weaker form of
reproducibility: can ``B``, given all of the code ``A`` wrote, run it and
produce the same result?

Often, the answer is no. Software environments are often difficult to specify
sufficiently well enough to replicate elsewhere.
Worse, much of the knowledge required to execute a particular workflow is often
in "soft" places: someone's brain; a text file that hasn't been updated in
several years; a sticky note next to someone's monitor.

By forcing all of the "science" part of running a simulation into the
:func:`Simulation.run` method, we effectively "self-document" what it means to
run a simulation. Now, that doesn't mean that the code won't be complex.
:func:`Simulation.run` could end up running an enormous amount of code.
But the "entry point" of the simulation is clear.

Similarly, forcing all of the configuration to be expressed in
:class:`Specification` means that we know where to find all of the input
parameters of a given simulation.
Since each :class:`Simulation` has a single :class:`Specification`, we know
exactly how it was configured to run.
A specification could be stored for a month, then loaded, turned into
a simulation, and run, and it would be just like it had been run a month ago.


Sanity
------

Research code is often written in a haphazard, fire-and-forget fashion. Everyone
has written code like this, and everyone promises to go back and clean it up
later. Sometimes they even do!

Simulacra fights this tendency by enforcing structure on our work. We cannot
inject new behavior by copying a script and editing a few lines. Instead, you
need to add new behavior in rigorous, well-controlled ways: add configuration
options to the :class:`Specification`, look for those options in the
:class:`Simulation` and respond appropriately.

This obviously takes more time, effort, and skill. But the end result is
software that is more powerful, flexible, and reusable than software produced
haphazardly. We are paying now, but we will be happier later.
