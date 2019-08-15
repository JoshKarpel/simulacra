#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simulacra documentation build configuration file, created by
# sphinx-quickstart on Mon May 15 20:53:07 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../src"))

autodoc_mock_imports = [
    "_tkinter"
]  # solves https://stackoverflow.com/questions/45484077/sphinx-autodoc-on-readthedocs-importerror-no-module-named-tkinter
import matplotlib

matplotlib.use("agg")

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # 'sphinx_autodoc_typehints',
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "simulacra"
copyright = "2017, Josh Karpel"
author = "Josh Karpel"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.2.0"
# The full version, including alpha/beta/rc tags.
release = "0.2.0"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "simulacradoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "simulacra.tex", "simulacra Documentation", "Josh Karpel", "manual")
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "simulacra", "simulacra Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "simulacra",
        "simulacra Documentation",
        author,
        "simulacra",
        "One line description of project.",
        "Miscellaneous",
    )
]

# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "http://docs.scipy.org/doc/numpy/": None,
    "http://matplotlib.org": None,
}

autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_default_flags = ["show-inheritance"]

napoleon_use_rtype = True

# MONKEY PATCH GET_DOC TO PUT PARAMETERS BEFORE ATTRIBUTES

import sphinx.ext.autodoc
import sphinx
from sphinx.util import force_decode
from sphinx.util.docstrings import prepare_docstring
from six import text_type


def get_doc(self, encoding=None, ignore=1):
    lines = getattr(self, "_new_docstrings", None)
    if lines is not None:
        return lines

    content = self.env.config.autoclass_content

    docstrings = []
    attrdocstring = self.get_attr(self.object, "__doc__", None)
    if attrdocstring:
        docstrings.append(attrdocstring)

    # for classes, what the "docstring" is can be controlled via a
    # config value; the default is only the class docstring
    if content in ("both", "init"):
        initdocstring = self.get_attr(
            self.get_attr(self.object, "__init__", None), "__doc__"
        )
        # for new-style classes, no __init__ means default __init__
        if initdocstring is not None and (
            initdocstring == object.__init__.__doc__
            or initdocstring.strip() == object.__init__.__doc__  # for pypy
        ):  # for !pypy
            initdocstring = None
        if not initdocstring:
            # try __new__
            initdocstring = self.get_attr(
                self.get_attr(self.object, "__new__", None), "__doc__"
            )
            # for new-style classes, no __new__ means default __new__
            if initdocstring is not None and (
                initdocstring == object.__new__.__doc__
                or initdocstring.strip() == object.__new__.__doc__  # for pypy
            ):  # for !pypy
                initdocstring = None
        if initdocstring:
            if content == "init":
                docstrings = [initdocstring]
            else:
                if len(docstrings) == 0 or "Attributes" not in docstrings[0]:
                    docstrings.append(initdocstring)
                else:
                    class_str = docstrings[0]

                    lines = class_str.split("\n")
                    for attributes_line, line in enumerate(lines):
                        if "Attributes" in line:
                            break

                    lines = (
                        lines[:attributes_line]
                        + [s[4:] for s in initdocstring.splitlines()]
                        + lines[attributes_line:]
                    )

                    docstrings = ["\n".join(lines)]
    doc = []
    for docstring in docstrings:
        if isinstance(docstring, text_type):
            doc.append(prepare_docstring(docstring, ignore))
        elif isinstance(docstring, str):  # this will not trigger on Py3
            doc.append(prepare_docstring(force_decode(docstring, encoding), ignore))
    return doc


sphinx.ext.autodoc.ClassDocumenter.get_doc = get_doc
