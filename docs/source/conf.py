import os
import sys

# -- Path setup --------------------------------------------------------------

# Add the project root to sys.path so autodoc can find SpikeSift
#sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'SpikeSift'
author = 'Vasileios Georgiadis'
copyright = '2025, Vasileios Georgiadis'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',      # Include docstrings from code
    'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx.ext.napoleon',     # Support for NumPy-style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
}

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'titles_only': True,
}

html_title = 'SpikeSift Documentation'
html_static_path = ['_static']
