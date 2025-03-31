import numpy as np
from ._core import SortedSegment
from ._core import segment_and_sort_spikes, find_optimal_assignment

class Recording:
    """
    Represents an extracellular recording stored in a flat binary file.

    This class manages metadata and provides efficient access to raw voltage data 
    for spike sorting. It assumes a flat binary layout with channels interleaved 
    sample-wise.

    Parameters
    ----------
    binary_file : str
        Path to the binary file containing the raw recording.
    data_type : dtype
        NumPy-compatible data type (e.g., ``float32``, ``int16``).
    probe_geometry : ndarray of shape (recording_channels, 2)
        Spatial coordinates (in micrometers) of each recording channel.
    sampling_frequency : float
        Sampling rate in Hz. Must be at least 1000.
    num_samples : int, optional
        Total number of samples to load. If omitted, the number is inferred 
        from file size and header.
    header : int, optional (default=0)
        Number of bytes to skip at the beginning of the file.
    sample_offset : int, optional (default=0)
        Number of samples to skip after the header.
    recording_offset : int, optional (default=0)
        Logical start time in samples, used for aligning or merging segments. 
        Does not affect how data are read.

    Warning
    -------
    - After creation, this object should be treated as read-only.
    - Binary layout must be flat and channel-interleaved sample-wise.
    - The order of channels in `probe_geometry` must match the binary file.
    """

    VALID_DATA_TYPES = ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64')

    def __init__(self, *, binary_file, data_type, probe_geometry, sampling_frequency, num_samples=None, header=0, sample_offset=0, recording_offset=0):    
        self.binary_file = binary_file
        self.data_type = data_type
        self.probe_geometry = probe_geometry
        self.sampling_frequency = sampling_frequency
        self.num_samples = num_samples
        self.header = header
        self.sample_offset = sample_offset
        self.recording_offset = recording_offset
        self.validate()

    def validate(self, *, verbose=False):
        """
        Finalizes setup and verifies recording consistency.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints a summary of the recording.

        Raises
        ------
        ValueError
            If any of the file, geometry, or offset parameters are invalid.

        Warning
        -------
        - This method is called automatically during spike sorting.
        - Manual calls are typically only necessary for debugging or inspection.
        """
        
        # Validate file accessibility
        try:
            file_map = np.memmap(self.binary_file, dtype=np.uint8, mode='r')
            self.file_size = file_map.size
        except Exception as e:
            raise ValueError(
                "Unable to read binary file "
                f"'{self.binary_file}': {e}"
            )

        # Validate and resolve data type
        try:
            self.data_type = np.dtype(self.data_type)
        except Exception as e:
            raise ValueError(
                "Unable to resolve data type "
                f"'{self.data_type}': {e}"
            )

        if self.data_type not in (np.dtype(x) for x in self.VALID_DATA_TYPES):
            raise ValueError(
                f"Invalid data type '{self.data_type}'. "
                f"Must be one of {self.VALID_DATA_TYPES}."
            )

        # Validate probe geometry
        if not isinstance(self.probe_geometry, np.ndarray) or self.probe_geometry.ndim != 2 or self.probe_geometry.shape[1] != 2:
            raise ValueError("probe_geometry must be a numpy array of shape (recording_channels, 2).")
        if self.probe_geometry.dtype.kind not in {'f', 'i', 'u'}:
            raise ValueError("probe_geometry must contain numeric values (float or int).")
        self.probe_geometry = self.probe_geometry.astype('float32')
        self.recording_channels = self.probe_geometry.shape[0]
        if self.recording_channels < 1:
            raise ValueError("No recording channels provided")

        # Validate sampling rate
        if not isinstance(self.sampling_frequency, (int, float, np.number)) or self.sampling_frequency < 1000:
            raise ValueError("sampling_frequency must be at least 1000 Hz.")
        self.samples_per_ms = int(self.sampling_frequency // 1000)

        # Validate offsets
        if not isinstance(self.header, (int, np.integer)) or self.header < 0:
            raise ValueError("header must be a non-negative integer.")
        if not isinstance(self.sample_offset, (int, np.integer)) or self.sample_offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        if not isinstance(self.recording_offset, (int, np.integer)):
            raise ValueError("recording_offset must be an integer.")

        # Check total offset
        self.bytes_per_sample = self.recording_channels * self.data_type.itemsize
        if self.header + self.sample_offset * self.bytes_per_sample >= self.file_size:
            raise ValueError("Offset exceeds file size; no valid data available.")
        available_bytes = self.file_size - (self.header + self.sample_offset * self.bytes_per_sample)

        # Validate num_samples or infer it
        if self.num_samples is None:
            if available_bytes % self.bytes_per_sample != 0:
                raise ValueError(
                    "File is not aligned to full samples "
                    "(specify num_samples if you want to proceed)."
                )
            self.num_samples = available_bytes // self.bytes_per_sample
        else:
            if not isinstance(self.num_samples, (int, np.integer)) or self.num_samples <= 0:
                raise ValueError(f"num_samples must be a positive integer, but got {self.num_samples}.")
            max_possible_samples = available_bytes // self.bytes_per_sample
            if self.num_samples > max_possible_samples:
                raise ValueError(
                    f"Requested num_samples ({self.num_samples}) "
                    f"exceeds available samples ({max_possible_samples}) "
                    "based on file size."
                )

        if verbose:
            print(
                "=" * 60 + "\n"
                " SpikeSift - Recording Info\n"
                + "-" * 60 + "\n"
                f" File path             : {self.binary_file}\n"
                f" Data type             : {self.data_type}\n"
                f" Header offset         : {self.header} bytes\n"
                f" Sample offset         : {self.sample_offset} samples\n"
                f" Recording offset      : {self.recording_offset} samples\n"
                f" Total duration        : {self.num_samples / self.sampling_frequency} seconds\n"
                f" Number of channels    : {self.recording_channels}\n"
                f" Sampling frequency    : {self.sampling_frequency} Hz\n"
                + "=" * 60 + "\n"
            )

    def read(self, *, start, num_samples):
        """
        Reads a segment of the binary recording.

        Parameters
        ----------
        start : int
            Sample index to begin reading, after accounting for ``header`` and ``sample_offset``.
        num_samples : int
            Number of consecutive samples to read.

        Returns
        -------
        ndarray, shape ``(num_samples, recording_channels)``
            Extracted signal data as a NumPy array.

        Warning
        -------
        - This method is intended for debugging and manual inspection only.
        - SpikeSift handles all necessary data access internally during sorting.
        """

        byte_offset = self.header + self.bytes_per_sample * (
            self.sample_offset + start
        )
        num_values = num_samples * self.recording_channels

        return np.fromfile(
            self.binary_file, 
            dtype = self.data_type, 
            offset = byte_offset, 
            count = num_values
        ).reshape(num_samples, self.recording_channels)


class SortedRecording:
    """
    Represents a fully sorted and drift-corrected extracellular recording.

    This class merges spike clusters across multiple independently sorted segments,
    and provides access to global spike times, amplitude vectors, and segment boundaries.

    Parameters
    ----------
    sorted_segments : list of SortedSegment (internal)
        List of sorted segments, each containing spike clusters and amplitude representations.
    assignment_chain : list of ndarray of shape ``(num_clusters,)``
        One-to-one mappings between adjacent segments.

        - Each array maps cluster indices from one segment to the next.
        - Unassigned entries are marked with -1.
    probe_geometry : ndarray of shape ``(recording_channels, 2)``
        2D electrode layout used for drift compensation.

    Warning
    -------
    - Do not modify ``sorted_segments``, ``assignment_chain``, or ``probe_geometry`` in place.
      They are shared across recordings and treated as immutable.
    """

    def __init__(self, *, sorted_segments, assignment_chain, probe_geometry):
        self.sorted_segments = sorted_segments
        self.assignment_chain = assignment_chain
        self.probe_geometry = probe_geometry

    def start_time(self):
        """
        Returns the global start time of the recording (in samples).

        Returns
        -------
        int
            Start time in samples.
        """
        return self.sorted_segments[0].start_time

    def end_time(self):
        """
        Returns the global end time of the recording (in samples).

        Returns
        -------
        int
            End time in samples.
        """
        last = self.sorted_segments[-1]
        return last.start_time + last.duration

    def segment_boundaries(self):
        """
        Returns start and end sample indices for all segments.

        Returns
        -------
        list of tuple
            List of ``(start_sample, end_sample)`` pairs, one per segment.
        """
        return [
            (seg.start_time, seg.start_time + seg.duration)
            for seg in self.sorted_segments
        ]
    
    def valid_cluster_id(self, cluster_id):
        """
        Checks whether a cluster ID is valid across the entire recording.

        Parameters
        ----------
        cluster_id : int
            The cluster ID to validate.

        Returns
        -------
        bool
            True if the cluster is consistently matched across all segments; False otherwise.

        Warning
        -------
        - A cluster is considered valid only if it is present in every segment of the recording.
        - Clusters that disappear or fragment in later segments will return False.
        """
        if not (
            isinstance(cluster_id, (int, np.integer)) 
            and 0 <= cluster_id < len(self.sorted_segments[0])
        ):
            return False

        for assignment in self.assignment_chain:
            cluster_id = assignment[cluster_id]
            if cluster_id == -1:
                return False
        return True
        
    def cluster_ids(self):
        """
        Returns all valid cluster IDs for this recording.

        Returns
        -------
        set of int
            Set of cluster IDs that are valid across the entire recording.

        Warning
        -------
        - IDs may refer to different units across different SortedRecording objects.
        - To compare clusters between recordings, use :func:`~spikesift.map_clusters`.
        """
        return {
            cid for cid in range(len(self.sorted_segments[0]))
            if self.valid_cluster_id(cid)
        }


    def spikes(self, cluster_id):
        """
        Returns spike times for the specified cluster.

        Parameters
        ----------
        cluster_id : int
            The cluster ID to retrieve.

        Returns
        -------
        ndarray
            1D NumPy array of spike times for the selected cluster.

        Raises
        ------
        ValueError
            If the cluster ID is not valid for this recording.

        Warning
        -------
        - Cluster IDs are only valid within this SortedRecording instance.
        - To avoid invalid lookups, use `.cluster_ids()` to retrieve the set of valid cluster IDs.
        """
        if not self.valid_cluster_id(cluster_id):
            raise ValueError(
                f"Cluster ID {cluster_id} "
                "is not valid for this SortedRecording."
            )

        first = self.sorted_segments[0]
        spike_times = [first.spike_clusters[cluster_id]]

        for i, assignment in enumerate(self.assignment_chain, start=1):
            cluster_id = assignment[cluster_id]
            spike_times.append(
                self.sorted_segments[i].spike_clusters[cluster_id]
            )

        return np.concatenate(spike_times)

    def amplitude_vectors(self, cluster_id):
        """
        Returns the amplitude vectors for a single cluster across all segments.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster.

        Returns
        -------
        ndarray of shape (num_segments, recording_channels)
            Amplitude vector for each segment.

        Raises
        ------
        ValueError
            If the cluster ID is not valid for this recording.

        Warning
        -------
        - Values reflect both spike-related activity and background 
          fluctuations, and may be nonzero even on channels where the neuron is inactive.
        """
        if not self.valid_cluster_id(cluster_id):
            raise ValueError(
                f"Cluster ID {cluster_id} "
                "is not valid for this SortedRecording."
            )

        first = self.sorted_segments[0]
        vectors = [first.amplitude_vectors[cluster_id]]

        for i, assignment in enumerate(self.assignment_chain, start=1):
            cluster_id = assignment[cluster_id]
            vectors.append(
                self.sorted_segments[i].amplitude_vectors[cluster_id]
            )

        return np.stack(vectors)

    def all_spikes(self):
        """
        Returns spike times for all valid clusters.

        Returns
        -------
        dict of int -> ndarray
            Dictionary mapping cluster IDs to spike times.
        """
        return {
            cid: self.spikes(cid) 
            for cid in self.cluster_ids()
        }

    def split_into_segments(self):
        """
        Splits the recording into its original unmerged segments.

        Returns
        -------
        list of SortedRecording
            Each entry corresponds to one original segment.
        """
        return [
            SortedRecording(
                sorted_segments=[seg],
                assignment_chain=[],
                probe_geometry=self.probe_geometry
            )
            for seg in self.sorted_segments
        ]
    
    def __len__(self):
        """
        Returns the number of valid clusters in the recording.

        Returns
        -------
        int
            Number of globally aligned clusters.
        """
        return len(self.cluster_ids())

def perform_spike_sorting(
    recording, 
    *, 
    min_segment_length = 10, 
    detection_sensitivity = 10, 
    min_spikes_per_cluster = 5, 
    merging_threshold = 0.4, 
    max_drift = 30, 
    detection_polarity = -1, 
    verbose = True
):
    """
    Performs complete spike sorting on an extracellular recording.

    Parameters
    ----------
    recording : Recording
        The input recording object.
    min_segment_length : float, optional (default=10)
        Minimum segment duration (in seconds) for adaptive segmentation.

        - Controls how the recording is partitioned
        - Must be at least 0.1 seconds
        - Values below 0.1 are automatically clipped
        - If the recording itself is shorter than this, it is processed as a single segment
    detection_sensitivity : float, optional (default=10)
        Multiplier for spike detection thresholds.

        - Must be positive
        - Higher values reduce false positives, but may miss weaker spikes
        - Lower values increase sensitivity, but may introduce noise
    min_spikes_per_cluster : float, optional (default=5)
        Minimum number of spikes required for a cluster to be considered valid.

        - Must be at least 2  
        - Values below 2 are silently clipped  
        - Although spike counts are integers, this threshold is treated as a float and compared directly
    merging_threshold : float, optional (default=0.4)
        Similarity threshold for merging clusters based on spatial waveform differences.

        - Must be between 0 and 1 (exclusive)
        - Higher values allow more aggressive merging
        - Lower values enforce stricter separation
    max_drift : float, optional (default=30)
        Maximum vertical shift (in micrometers) used for aligning clusters across segments.

        - Must be non-negative
        - Internally rounded to the nearest multiple of 5
        - Larger values enable alignment over larger displacements
    detection_polarity : float, optional (default=-1)
        Scalar applied to the signal prior to spike detection.
        
        - Use -1.0 to detect negative-going spikes (default)
        - Use +1.0 to detect positive-going spikes
        - Any other nonzero value is allowed; only the sign affects detection
    verbose : bool, optional (default=True)
        If True, displays progress bar and recording information.

    Returns
    -------
    SortedRecording
        A fully sorted recording, including spike times, cluster identities, and amplitude vectors.

    Raises
    ------
    ValueError
        If any input parameter is invalid or improperly typed.
    
    Warning
    -------
    - Recordings shorter than 10 milliseconds cannot be processed and will raise an error.
    - SpikeSift requires at least 4 channels for spike sorting.
    """

    # Validate recording object
    if not isinstance(recording, Recording):
        raise ValueError("recording must be an instance of Recording.")
    recording.validate(verbose = verbose)
    if recording.num_samples / recording.samples_per_ms < 10:
        raise ValueError(
            "Recording is too short for reliable spike sorting "
            f"({recording.num_samples / recording.sampling_frequency * 1000:.1f} ms total). "
            "At least 10 ms are required."
        )
    if recording.recording_channels < 4:
        raise ValueError(
            f"Recording has only {recording.recording_channels} channels. "
            "SpikeSift requires at least 4 channels for spike sorting."
        )
    
    # Validate minimum segment length
    if not isinstance(min_segment_length, (int, float, np.number)):
        raise ValueError(f"min_segment_length must be a number, but got {min_segment_length}.")
    min_segment_length = max(min_segment_length, 0.1)
    min_segment_length = int(round(min_segment_length * 1000))  # convert to ms

    # Validate sensitivity
    if not isinstance(detection_sensitivity, (int, float, np.number)) or detection_sensitivity <= 0:
        raise ValueError(f"detection_sensitivity must be a positive number, but got {detection_sensitivity}.")

    # Validate cluster size
    if not isinstance(min_spikes_per_cluster, (int, float, np.number)):
        raise ValueError(f"min_spikes_per_cluster must be a number, but got {min_spikes_per_cluster}.")
    min_spikes_per_cluster = max(min_spikes_per_cluster, 2)

    # Validate merging threshold
    if not isinstance(merging_threshold, (int, float, np.number)) or not (0 < merging_threshold < 1):
        raise ValueError(f"merging_threshold must be between 0 and 1 (exclusive), but got {merging_threshold}.")

    # Validate and snap drift to 5um resolution
    if not isinstance(max_drift, (int, float, np.number)) or max_drift < 0:
        raise ValueError(f"max_drift must be a non-negative float or int, but got {max_drift}.")
    max_drift = int(round(max_drift / 5)) * 5
    
    # Validate detection multiplier
    if not isinstance(detection_polarity, (int, float, np.number)) or detection_polarity == 0:
        raise ValueError(f"detection_polarity must be a nonzero float, but got {detection_polarity}.")

    # Segment the recording and sort spikes
    sorted_segments = segment_and_sort_spikes(
        recording,
        min_segment_length = min_segment_length,
        detection_sensitivity = detection_sensitivity,
        min_spikes_per_cluster = min_spikes_per_cluster,
        merging_threshold = merging_threshold,
        detection_polarity = detection_polarity,
        verbose = verbose
    )

    # Align spike clusters across segments with simulated drift
    assignment_chain = []
    for i in range(len(sorted_segments) - 1):
        assignment = find_optimal_assignment(
            sorted_segments[i].amplitude_vectors,
            sorted_segments[i + 1].amplitude_vectors,
            probe_geometry = recording.probe_geometry,
            max_drift = max_drift
        )
        assignment_chain.append(assignment)

    # Return fully sorted, drift-aligned result
    return SortedRecording(
        sorted_segments = sorted_segments,
        assignment_chain = assignment_chain,
        probe_geometry = recording.probe_geometry.copy()
    )

def merge_recordings(
    sorted_recordings,
    *,
    max_drift = 30
):
    """
    Aligns and merges multiple independently sorted recordings into a unified result.

    Parameters
    ----------
    sorted_recordings : list of SortedRecording
        List of independently sorted recordings to be merged. Each entry must:

        - Contain at least one valid segment
        - Use the same probe geometry
        - Be sorted in time and have non-overlapping segments

    max_drift : float, optional (default=30)
        Maximum vertical shift (in micrometers) allowed when aligning clusters across segments.

        - Must be non-negative
        - Internally rounded to the nearest multiple of 5
        - Higher values allow alignment over larger displacements

    Returns
    -------
    SortedRecording
        A single merged recording containing all aligned spike clusters.

    Raises
    ------
    ValueError
        If the input list is empty, contains invalid types, includes inconsistent geometries,
        or includes overlapping segment time ranges.

    Warning
    -------
    - This function assumes all inputs were produced by SpikeSift and remain unmodified.
    """

    # Validate max_drift
    if not isinstance(max_drift, (int, float, np.number)) or max_drift < 0:
        raise ValueError(f"`max_drift` must be a non-negative float or int (got {max_drift}).")
    max_drift = int(round(max_drift / 5)) * 5

    # Validate input list
    if not isinstance(sorted_recordings, list) or len(sorted_recordings) == 0:
        raise ValueError("`sorted_recordings` must be a non-empty list of SortedRecording objects.")

    # Validate geometry
    def validate_geometry(arr, label):
        if not (
            isinstance(arr, np.ndarray)
            and arr.dtype == np.float32
            and arr.ndim == 2
            and arr.shape[1] == 2
            and arr.flags['C_CONTIGUOUS']
        ):
            raise ValueError(
                f"{label} must be a C-contiguous float32 array of shape (n, 2). "
                "This may indicate that one of the SortedRecording objects or their segments was modified manually. "
                "Note: segments are shared between recordings and should be treated as immutable."
            )

    for i, rec in enumerate(sorted_recordings):
        if not isinstance(rec, SortedRecording):
            raise ValueError(f"Item {i} in `sorted_recordings` is not a SortedRecording.")
        validate_geometry(rec.probe_geometry, f"probe_geometry of recording {i}")

    probe_geometry = sorted_recordings[0].probe_geometry
    num_channels = probe_geometry.shape[0]

    for i in range(1, len(sorted_recordings)):
        if not np.array_equal(sorted_recordings[i].probe_geometry, probe_geometry):
            raise ValueError(f"probe_geometry of recording {i} does not match earlier recordings.")
        if sorted_recordings[i - 1].end_time() > sorted_recordings[i].start_time():
            raise ValueError(
                f"`sorted_recordings` must be ordered and non-overlapping in time: "
                f"recording {i - 1} ends at {sorted_recordings[i - 1].end_time()}, "
                f"which overlaps with recording {i} starting at {sorted_recordings[i].start_time()}."
            )

    # Validate amplitude vectors
    def validate_amplitude_vectors(vecs, label):
        if not (
            isinstance(vecs, np.ndarray)
            and vecs.dtype == np.float32
            and vecs.ndim == 2
            and vecs.shape[1] == num_channels
            and vecs.flags['C_CONTIGUOUS']
        ):
            raise ValueError(
                f"{label} must be a C-contiguous float32 array of shape (num_clusters, {num_channels}). "
                "This may indicate that one of the SortedRecording objects or their segments was modified manually. "
                "Note: segments are shared between recordings and should be treated as immutable."
            )

    # Flatten all segments
    sorted_segments = [
        segment for rec in sorted_recordings
        for segment in rec.sorted_segments
    ]

    # Merge assignment chains
    updated_assignment_chain = []

    for i, rec in enumerate(sorted_recordings):
        updated_assignment_chain.extend(rec.assignment_chain)

        # Bridge between adjacent recordings
        if i < len(sorted_recordings) - 1:
            seg_A = rec.sorted_segments[-1]
            seg_B = sorted_recordings[i + 1].sorted_segments[0]

            validate_amplitude_vectors(
                seg_A.amplitude_vectors, 
                f"Amplitude vectors of last segment in recording {i}"
            )
            validate_amplitude_vectors(
                seg_B.amplitude_vectors, 
                f"Amplitude vectors of first segment in recording {i + 1}"
            )

            bridge_assignment = find_optimal_assignment(
                seg_A.amplitude_vectors,
                seg_B.amplitude_vectors,
                probe_geometry=probe_geometry,
                max_drift=max_drift
            )
            updated_assignment_chain.append(bridge_assignment)

    return SortedRecording(
        sorted_segments=sorted_segments,
        assignment_chain=updated_assignment_chain,
        probe_geometry=probe_geometry
    )

def map_clusters(
    source,
    target,
    *,
    max_drift=30
):
    """
    Computes a one-to-one mapping from clusters in ``source`` to their counterparts in ``target``.

    Parameters
    ----------
    source : SortedRecording
        First sorted recording to compare.

    target : SortedRecording
        Second sorted recording to compare.

    max_drift : float, optional (default=30)
        Maximum vertical displacement (in micrometers) used during alignment.

        - Must be non-negative
        - Internally rounded to the nearest multiple of 5
        - Higher values permit alignment across larger drift magnitudes

    Returns
    -------
    dict of int -> int
        Mapping from cluster IDs in ``source`` to corresponding cluster IDs in ``target``.
        Only valid, unambiguous one-to-one matches are included.

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible (e.g., mismatched geometry).

    Warning
    -------
    - This function assumes that both ``source`` and ``target`` were generated using SpikeSift
      and have not been manually modified.
    """

    # Validate and round drift threshold
    if not isinstance(max_drift, (int, float, np.number)) or max_drift < 0:
        raise ValueError(f"`max_drift` must be a non-negative float or int (got {max_drift}).")
    max_drift = int(round(max_drift / 5)) * 5

    # Validate input types
    if not isinstance(source, SortedRecording):
        raise ValueError("`source` must be a SortedRecording instance.")
    if not isinstance(target, SortedRecording):
        raise ValueError("`target` must be a SortedRecording instance.")

    def validate_geometry(arr, label):
        if not (
            isinstance(arr, np.ndarray)
            and arr.dtype == np.float32
            and arr.ndim == 2
            and arr.shape[1] == 2
            and arr.flags['C_CONTIGUOUS']
        ):
            raise ValueError(
                f"{label} must be a C-contiguous float32 array of shape (n, 2). "
                "This may indicate that one of the SortedRecording objects or their segments was modified manually. "
                "Note: segments are shared between recordings and should be treated as immutable."
            )

    validate_geometry(
        source.probe_geometry, 
        "source.probe_geometry"
    )
    validate_geometry(
        target.probe_geometry, 
        "target.probe_geometry"
    )

    if not np.array_equal(source.probe_geometry, target.probe_geometry):
        raise ValueError("`probe_geometry` must match exactly between source and target.")

    # Validate amplitude vectors
    probe_geometry = source.probe_geometry
    num_channels = probe_geometry.shape[0]
    source_amplitudes = source.sorted_segments[0].amplitude_vectors
    target_amplitudes = target.sorted_segments[0].amplitude_vectors

    def validate_amplitude_vectors(vecs, label):
        if not (
            isinstance(vecs, np.ndarray)
            and vecs.dtype == np.float32
            and vecs.ndim == 2
            and vecs.shape[1] == num_channels
            and vecs.flags['C_CONTIGUOUS']
        ):
            raise ValueError(
                f"{label} must be a C-contiguous float32 array of shape (num_clusters, {num_channels}). "
                "This may indicate that one of the SortedRecording objects or their segments was modified manually. "
                "Note: segments are shared between recordings and should be treated as immutable."
            )

    validate_amplitude_vectors(
        source_amplitudes,
        "source.amplitude_vectors"
    )
    validate_amplitude_vectors(
        target_amplitudes, 
        "target.amplitude_vectors"
    )

    # Perform optimal assignment using amplitude-based alignment
    assignment = find_optimal_assignment(
        source_amplitudes,
        target_amplitudes,
        probe_geometry=probe_geometry,
        max_drift=max_drift
    )

    # Construct cluster ID mapping
    source_ids = source.cluster_ids()
    target_ids = target.cluster_ids()

    cluster_map = {}
    for i, j in enumerate(assignment):
        if i in source_ids and j in target_ids:
            cluster_map[i] = j

    return cluster_map
