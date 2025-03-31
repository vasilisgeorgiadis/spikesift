.. _installation:

Installation
============

SpikeSift supports Python 3.7 and later and can be installed via `pip` or directly from source.

Installing with pip
-------------------

To install the latest stable release from PyPI:

.. code-block:: bash

   pip install spikesift

This will automatically install all required dependencies, including:

- `numpy`  
- `scipy`  
- `cython`  
- `tqdm` (used for progress bars)

Installing from Source
----------------------

To install the latest development version:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/vasilisgeorgiadis/spikesift.git
      cd spikesift

2. Install in editable mode (compiles the Cython extensions and links the code locally):

   .. code-block:: bash

      pip install -e .

Troubleshooting
---------------

If installation fails due to build issues or missing compilers:

1. Make sure your build environment is up to date:

   .. code-block:: bash

      pip install --upgrade pip setuptools wheel

2. Ensure a working C++ compiler is available on your system.

3. Retry the installation.

.. note::

    On Windows, make sure you have the **Microsoft C++ Build Tools** installed.  
    This is required to compile the Cython extensions. You can install them from:

    - https://visualstudio.microsoft.com/visual-cpp-build-tools/

Verifying Installation
----------------------

To verify that SpikeSift was installed correctly:

.. code-block:: python

   import spikesift
   print(spikesift.__version__)

If this runs without error, SpikeSift is ready to use.
