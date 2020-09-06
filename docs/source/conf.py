# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os
import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'cONNXr'
copyright = '2020, alrevuelta, nopeslide'
author = 'alrevuelta, nopeslide'

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['recommonmark']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_js_files = ['js/expand_tabs.js']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#html_logo = 'img/logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False