.. _user_guide:

User Guide
==========

This guide walks through how to use SpikeSift to perform spike sorting and interpret the results.  
It introduces the main API functions and explains how to access results.

Running Spike Sorting
---------------------

The primary entry point is :func:`~spikesift.perform_spike_sorting`. It accepts a :class:`~spikesift.Recording` object and returns a sorted and drift-corrected :class:`~spikesift.core.SortedRecording`.

.. code-block:: python

    from spikesift import Recording, perform_spike_sorting

    probe = np.load("probe_layout.npy")  # Example probe layout

    recording = Recording(
        binary_file="recording.bin",
        data_type="int16",
        probe_geometry=probe,
        sampling_frequency=30000
    )

    result = perform_spike_sorting(recording)

To control sorting behavior, you can adjust optional parameters such as sensitivity, minimum segment duration, and merging thresholds.  
Details are provided in :ref:`parameter_tuning`.

Working with the Sorted Output
------------------------------

The output of sorting is a :class:`~spikesift.core.SortedRecording` object.  
This object provides access to globally aligned spike clusters, their spike times, and amplitude profiles across segments.

Key methods include:

- ``result.cluster_ids()``  
  Returns the set of valid ``cluster_ids`` in this recording as a Python `set`.

  .. code-block:: python

     for cluster_id in result.cluster_ids():
         print("Cluster", cluster_id)

- ``result.valid_cluster_id(cluster_id)``  
  Checks if a ``cluster_id`` is globally valid (present in every segment).

  .. code-block:: python

     if result.valid_cluster_id(5):
         print("Cluster 5 is valid")

- ``result.spikes(cluster_id)``  
  Returns spike times for the given cluster. Raises an error if ``cluster_id`` is invalid.

  .. code-block:: python

     if cluster_id in result.cluster_ids():
         spikes = result.spikes(cluster_id)

- ``result.amplitude_vectors(cluster_id)``  
  Returns a 2D array ``(num_segments, recording_channels)`` showing the average waveform structure of the cluster over time.

  .. code-block:: python

     if cluster_id in result.cluster_ids():
         amps = result.amplitude_vectors(cluster_id)

  .. warning::
    
    - Values reflect both spike-related activity and background fluctuations, and may be nonzero even on channels where the neuron is inactive.

- ``result.all_spikes()``  
  Returns a dictionary mapping each valid ``cluster_id`` to its spike times.

  .. code-block:: python

     all_spikes = result.all_spikes()
     for cluster_id in all_spikes:
         print("Cluster", cluster_id, "has", len(all_spikes[cluster_id]), "spikes")

- ``result.segment_boundaries()``  
  Returns a list of ``(start_sample, end_sample)`` pairs for each aligned segment.  
  Useful for visualizing where the recording was split.

  .. code-block:: python

     boundaries = result.segment_boundaries()
     for i, (start, end) in enumerate(boundaries):
         print("Segment",i,"starts at",start,"and ends at",end)

- ``result.split_into_segments()``  
  Returns a list of one-segment :class:`~spikesift.core.SortedRecording` objects, each corresponding to an original unmerged segment.

  .. code-block:: python

     segments = result.split_into_segments()

- ``len(result)``  
  Returns the number of valid clusters in the full recording.

  .. code-block:: python

     print("Number of valid clusters:", len(result))

Merging Results from Multiple Sorts
-----------------------------------

If you sort long recordings in chunks, you can merge them using :func:`~spikesift.merge_recordings`.

.. code-block:: python

    from spikesift import merge_recordings

    merged = merge_recordings([result1, result2])

This merges all segments and aligns clusters across them.  
All outputs remain consistent with the same interface as described above, 
making this useful when sorting in parallel or appending new files.

Comparing Clusters Across Recordings
------------------------------------

``cluster_ids`` are only valid within the same :class:`~spikesift.core.SortedRecording` object.  
If you split or re-merge recordings, use :func:`~spikesift.map_clusters` to compare cluster identities:

.. code-block:: python

    from spikesift import map_clusters

    cluster_map = map_clusters(source=result1, target=result2)

    # Example: get spikes from result2 matching cluster 4 in result1
    if 4 in cluster_map:
        spikes = result2.spikes(cluster_map[4])

Cluster mappings are based on waveform similarity and drift-aware alignment.  
They allow flexible comparisons between segments, progressive recordings, and independently processed results.

Segment and Time Access
-----------------------

- ``result.start_time()`` and ``result.end_time()``  
  Give the full time range of the sorted recording.

  .. code-block:: python

     time_in_samples = result.start_time()
     time_in_seconds = time_in_samples / recording.sampling_frequency

- ``result.segment_boundaries()``  
  Returns where each segment begins and ends in the timeline.

These can help relate clusters to time intervals, organize results chronologically, or visualize activity across different periods of the recording.

.. warning::

   - Always use methods like ``spikes()``, ``cluster_ids()``, or ``amplitude_vectors()`` to access results.
   - Do not modify any part of a :class:`~spikesift.core.SortedRecording` in-place.
   - Modifying internal attributes may lead to corrupted or invalid results.

For full examples and scenario-based usage, see the :ref:`example_usage` section.

.. toctree::
   :hidden:

   user_guide/recording_setup
   user_guide/parameter_tuning
