.. _recording_setup:

Recording Setup
===============

This section explains how to define a :class:`~spikesift.Recording` object, the primary input required to run spike sorting.
SpikeSift reads extracellular recordings directly from disk using memory-mapped access.  
This requires a binary file with a well-defined structure and known probe geometry.

What format should the data have?
---------------------------------

The binary file must represent a 2D array of shape ``(num_samples, num_channels)``.

Data must be stored in sample-major (channel-interleaved) order --- 
all channel values for the first sample come first, followed by all channel values for the second sample, and so on.

Each value must match the specified ``data_type``. Supported data types include:

- ``int8``, ``uint8``
- ``int16``, ``uint16``
- ``int32``, ``uint32``
- ``int64``, ``uint64``
- ``float32``, ``float64``

The file must contain only the signal data --- no headers, markers, or extra content --- 
unless explicitly handled via parameters like ``header`` or ``sample_offset``.

What does the probe geometry represent?
---------------------------------------

The ``probe_geometry`` defines the **physical layout** of the recording sites.  
It must be a NumPy array of shape ``(num_channels, 2)``, where each row gives the ``(x, y)`` position  
(in micrometers) of one channel on the probe.

This layout is used for:

- Grouping nearby channels during waveform extraction  
- Tracking and aligning drifting neurons across time.

.. warning::

   - The order of channels in ``probe_geometry`` must exactly match the order in the binary file.  
     If your acquisition software saves channels in a different order, you must permute the probe accordingly.

How to define the probe layout
------------------------------

You can define the probe manually or load it from a file.

Example: a vertical probe with 16 channels spaced 20 micrometers apart:

.. code-block:: python

   import numpy as np
   from spikesift import Recording

   probe = np.array([[0, i * 20] for i in range(16)], dtype=np.float32)

   recording = Recording(
       binary_file="recording.dat",
       data_type="int16",
       probe_geometry=probe,
       sampling_frequency=20000
   )

.. note::

   - If your probe is stored in `.prb`, `.json`, `.nwb`, or other formats,  
     consider using the `probeinterface` library to convert it into a NumPy array.

How to handle headers and padding
---------------------------------

If the binary file contains metadata or padding before the actual data, use:

- ``header``: number of **bytes** to skip at the beginning of the file  
- ``sample_offset``: number of **samples** (not bytes) to skip after the header

These options help SpikeSift locate the start of the valid signal.

Example: skip a 1024-byte header and 1000 samples of padding:

.. code-block:: python

   recording = Recording(
       binary_file="recording_with_header.dat",
       data_type="float32",
       probe_geometry=probe,
       sampling_frequency=30000,
       header=1024,
       sample_offset=1000
   )

Other useful parameters
-----------------------

Additional arguments provide greater flexibility:

- ``num_samples``:  
  Restricts how many samples to read (after applying header and sample offset).  
  Useful if the file contains trailing padding or you only want to sort a portion.

- ``recording_offset``:  
  Sets the logical start time (in samples) of this recording within a larger session.  
  This does not affect how data are read — only how spike times are reported.
  This is essential for aligning multiple recordings.

Example: read 5 seconds of data and report spike times as if the recording started at 60 seconds:

.. code-block:: python

   recording = Recording(
       binary_file="block.dat",
       data_type="int16",
       probe_geometry=probe,
       sampling_frequency=30000,
       num_samples=5 * 30000,          # only read 5 seconds
       recording_offset=60 * 30000     # treat this as starting at t = 60s
   )

.. note::

   - ``recording_offset`` ensures that spike times from separate files or blocks  
     remain correctly aligned in time when merging them.