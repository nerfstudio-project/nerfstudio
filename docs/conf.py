# pylint: skip-file
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

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath("./_pygments"))
import nerfstudio.configs.base_config

# -- Project information -----------------------------------------------------

project = "nerfstudio"
copyright = "2022, nerfstudio Team"
author = "nerfstudio Team"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxarg.ext",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinxemoji.sphinxemoji",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.mathjax",
    "sphinxext.opengraph",
    "sphinx.ext.viewcode",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Needed for interactive plotly in notebooks
html_js_files = [
    "require.min.js",
    "custom.js",
]

# -- MYST configs -----------------------------------------------------------

# To enable admonitions:
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "html_image"]


# -- Options for open graph -------------------------------------------------

ogp_site_url = "http://docs.nerf.studio/"
ogp_image = "https://assets.nerf.studio/opg.png"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "nerfstudio"

autosectionlabel_prefix_document = True

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#d34600",
        "color-brand-content": "#ff6f00",
    },
    "dark_css_variables": {
        "color-brand-primary": "#fdd06c",
        "color-brand-content": "##fea96a",
    },
    "light_logo": "imgs/logo.png",
    "dark_logo": "imgs/logo-dark.png",
}

# -- Code block theme --------------------------------------------------------

pygments_style = "style.NerfstudioStyleLight"
pygments_dark_style = "style.NerfstudioStyleDark"

# -- Napoleon settings -------------------------------------------------------

# Settings for parsing non-sphinx style docstrings. We use Google style in this
# project.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- MYSTNB -----------------------------------------------------------------

suppress_warnings = ["mystnb.unknown_mime_type", "myst.header"]
nb_execution_mode = "off"
