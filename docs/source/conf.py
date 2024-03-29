# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_rtd_theme                                                 # noqa

sys.path.insert(0, os.path.abspath('../../src'))

from dinjo import VERSION

# -- Project information -----------------------------------------------------

project = 'DINJO'
copyright = '2021, Juan Esteban Aristizabal-Zuluaga and FEnFiSDi'
author = 'Juan Esteban Aristizabal-Zuluaga and FEnFiSDi'

dinjo_version = VERSION.split('.')

version = '.'.join(dinjo_version[:2])
release = '.'.join(dinjo_version[:3])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'private-members':
        '_seir_model, '
        '_seirv_model, '
        '_seirv_fixed, '
        '_sir_model',
    'special-members': '__init__',
    'inherited-members': False,
    'show-inheritance': False,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Some latex configurations, therea re many other options...
# If you want internal links in your pdf, run twise 'make latexpdf'.
latex_elements = {
    'papersize': 'a4paper,landscape',
}
