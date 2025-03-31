.. _example_usage:

Example Usage
=============

This walkthrough demonstrates how to use SpikeSift across a variety of common scenarios.  
It begins with basic spike sorting and gradually introduces parallel processing, handling split files, 
merging strategies, transient neuron tracking, and visualization.

All examples assume the probe layout has been loaded as a NumPy array:

.. code-block:: python

    import numpy as np
    from spikesift import Recording, perform_spike_sorting

    probe = np.load("probe.npy")

Performing Spike Sorting on a Single File
-----------------------------------------

To sort a single continuous recording:

.. code-block:: python

    recording = Recording(
        binary_file="recording.bin",
        data_type="int16",
        probe_geometry=probe,
        sampling_frequency=30000
    )

    result = perform_spike_sorting(recording)

Parallel Spike Sorting Across Multiple Cores
--------------------------------------------

For long recordings, divide the file into blocks and sort them in parallel.  
Each block is treated as an independent :class:`~spikesift.Recording` object.  
Once sorted, all results can be merged into a single unified output.

.. code-block:: python

    from joblib import Parallel, delayed
    from spikesift import merge_recordings

    # Split the recording into blocks
    num_blocks = 4
    samples_per_block = recording.num_samples // num_blocks

    recordings = [
        Recording(
            binary_file="recording.bin",
            data_type="int16",
            probe_geometry=probe,
            sampling_frequency=30000,
            sample_offset=i * samples_per_block,       # skip samples from previous blocks
            recording_offset=i * samples_per_block,    # align spike times across blocks
            num_samples=samples_per_block
        )
        for i in range(num_blocks)
    ]

    # Sort each block in parallel
    sorted_blocks = Parallel(n_jobs=4)(
        delayed(perform_spike_sorting)(r) for r in recordings
    )

    # Merge results into a single SortedRecording
    result = merge_recordings(sorted_blocks)

Adding a Second File
--------------------

If your recording spans multiple binary files, you can sort them separately and merge the results later.  
This avoids file concatenation and supports progressive analysis.

Assuming the first file has already been sorted:

.. code-block:: python

    # Create a Recording for the second file
    recording2 = Recording(
        binary_file="recording_part2.bin",
        data_type="int16",
        probe_geometry=probe,
        sampling_frequency=30000,
        recording_offset=result.end_time() # logical continuation
    )

    # Sort the second file
    result2 = perform_spike_sorting(recording2)

    # Merge both results into a unified recording
    result = merge_recordings([result, result2])

.. warning::

    - If the probe was repositioned or displaced between files,  you may need to increase 
      ``max_drift`` when calling :func:`~spikesift.merge_recordings` to allow clusters to be aligned correctly.

Adding a Third File with Modified Scaling
-----------------------------------------

If a subsequent file was saved in a different scale (e.g., after applying a linear transform like ``ax + b``), 
you can still merge it safely, as long as the transformation is consistent across channels.

You simply adjust the ``detection_polarity`` during spike sorting to account for the scale change.

.. code-block:: python

    # Suppose the third file was saved as float32 after applying: new_signal = a * original + b
    a = 0.1  # scaling factor
    b = 1.0  # offset

    # Since spike detection is based on amplitude, we undo the scaling during sorting
    recording3 = Recording(
        binary_file="recording_part3.bin",
        data_type="float32",
        probe_geometry=probe,
        sampling_frequency=30000,
        recording_offset=result.end_time()
    )

    # Apply inverse scaling during detection to match previous recordings
    result3 = perform_spike_sorting(
        recording3,
        detection_polarity = -1.0 / a  # Use a negative inverse for negative spikes (default behavior)
    )

    # Merge into final result
    result = merge_recordings([result, result3])

.. note::

    - The offset b does not affect spike sorting and does not need to be corrected.

Undoing or Refining a Merge
---------------------------

If you suspect over-merging, you can split the recording back into segments 
and selectively re-merge without rerunning sorting.

.. code-block:: python

    # Split the merged result into individual segments
    segments = result.split_into_segments()

    # Quickly inspect the number of clusters in each segment
    for i, seg in enumerate(segments):
        print(f"Segment {i} has {len(seg)} clusters")

    # Now you can selectively re-merge them
    # For example, merge only the first two segments:
    partial = merge_recordings(segments[:2])

    # Or manually decide which segments to exclude or group
    # Just ensure they remain in chronological order and do not overlap

Matching Clusters Across Recordings
-----------------------------------

Cluster IDs are only meaningful within a single :class:`~spikesift.core.SortedRecording`.  
If you reprocess, merge, or split a recording, IDs will not remain consistent.  
Use :func:`~spikesift.map_clusters` to align cluster identities between recordings:

.. code-block:: python

    from spikesift import map_clusters

    # Suppose you previously saved a result:
    reference = result # the earlier SortedRecording (before changes)

    # Now you have a new result after further merging or editing:
    new_result = merge_recordings([...]) # or another SortedRecording object

    # Compute cluster-to-cluster alignment between them:
    cluster_map = map_clusters(reference, new_result)

    # This returns a dictionary like:
    # {0: 3, 1: 7, 2: 5, ...}
    # Meaning cluster 0 in reference matches cluster 3 in new_result, etc.

    reference_cluster_id = 2

    if reference_cluster_id in cluster_map:
        aligned_id = cluster_map[reference_cluster_id]
        spikes = new_result.spikes(aligned_id)
    else:
        print("Cluster not matched in the new recording.")

.. warning::

    - If the two recordings are far apart in time, you may need to increase 
      ``max_drift`` when calling :func:`~spikesift.map_clusters` to allow proper alignment.

Tracking a Missing Cluster Across Segments
------------------------------------------

If a cluster from the original recording disappears after merging,  
it may have been lost in one or more segments.  
You can align segments individually to find where it still appears:

.. code-block:: python

    # Split the new recording into segments for individual inspection
    segments = new_result.split_into_segments()

    # Align the reference cluster to each segment individually
    reference_cluster_id = 2

    for i, segment in enumerate(segments):
        cluster_map = map_clusters(reference, segment)
        if reference_cluster_id in cluster_map:
            print(f"Cluster matched in segment {i} as ID {cluster_map[reference_cluster_id]}")

Tracking Transient Neurons in Long Recordings
---------------------------------------------

In long recordings, some neurons may fire only intermittently --- appearing in some segments but not others.
To track these transient neurons, you can use a short, representative reference and compare new segments individually:

.. code-block:: python

    # Assume this is your reference: a sorted segment with reliable clusters
    reference = perform_spike_sorting(short_recording)

    # Sort a longer recording and split it into smaller segments
    new_result = perform_spike_sorting(long_recording)
    segments = new_result.split_into_segments()

    # Track each cluster from the reference across the new segments
    for cluster_id in reference.cluster_ids():
        spikes = reference.spikes(cluster_id)

        # Compare against each segment individually
        for seg in segments:
            cluster_map = map_clusters(reference, seg, max_drift=50)  # adjust if needed
            if cluster_id in cluster_map:
                matched_id = cluster_map[cluster_id]
                spikes = np.concatenate([spikes, seg.spikes(matched_id)])

        print(f"Cluster {cluster_id} found in {len(spikes)} samples")

Visualizing Sorted Spikes
-------------------------

To assess the result visually, you can generate a simple raster plot showing all detected spikes.
Each line indicates a spike time, and the background alternates by segment:

.. code-block:: python

    import matplotlib.pyplot as plt

    sf = recording.sampling_frequency  # samples per second
    cluster_ids = list(result.cluster_ids())

    # Plot spike times for each cluster
    for idx, cluster_id in enumerate(cluster_ids):
        spike_times = result.spikes(cluster_id) / sf # convert to seconds
        plt.vlines(spike_times, idx - 0.05, idx + 0.05, color='black')

    # Highlight segment boundaries (in seconds)
    colors = ['lightblue', 'lightgreen']
    for i, (start, end) in enumerate(result.segment_boundaries()):
        plt.axvspan(start / sf, end / sf, color=colors[i % 2], alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Cluster ID")
    plt.title("Spike Raster Plot")
    plt.yticks(np.arange(len(cluster_ids)), cluster_ids)
    plt.tight_layout()
    plt.show()
