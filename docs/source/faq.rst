Frequently Asked Questions
==========================

How do I install Simulacra?
---------------------------

Installing Simulacra is easy to do using ``pip``.

::

    $ pip install simulacra

I recommend using `conda <https://conda.io/docs/intro.html>`_ for general Python package management, although Simulacra itself must be installed using pip.


How do I get the most recent version of Simulacra?
--------------------------------------------------

If you want to get the absolute latest version of Simulacra, you need to pull the source code from GitHub.
First, install `git <https://git-scm.com/>`_.
Then, move to the directory where you want to store the code and do

::

    $ git clone https://github.com/JoshKarpel/simulacra

To update this version in the future, go to the directory where you stored it and do

::

    $ git pull

This will get you a copy of the code, but you won't be able to import it while running Python.
To install the version of Simulacra you just downloaded, do

::

    $ pip install -e path_to_simulacra_dir

The ``-e`` option installs the Simulacra package in editable mode: you can import it as if it was a normally-installed package, but with whatever changes you make to the source code locally.
