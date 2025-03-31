import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ._cython.preprocessing import apply_bandpass_filter, transpose_and_compute_minima, compute_drift_heuristic
from ._cython.detection import compute_amplitude_scores, detect_spike_peaks, extract_spike_features, extract_waveforms
from ._cython.clustering import compute_inter_channel_differences, perform_hierarchical_clustering, perform_final_clustering 
from ._cython.subtraction import compute_average_spike_amplitude, subtract_average_waveform, update_spike_statistics
from ._cython.alignment import simulate_drift

class SortedSegment:
    """
    Represents a single spike-sorted segment of an extracellular recording.

    This class:
    - Stores metadata and results for one contiguous segment of the recording.
    - Holds the spike times and amplitude-based representations for detected neurons.
    - Acts as the output unit for intermediate and final sorting stages.

    Attributes
    ----------
    start_time : int
        Start time of the segment (in samples from the beginning of the recording).
    duration : int
        Duration of the segment in samples.
    spike_clusters : list of ndarray
        List of spike time arrays, one per detected cluster. Each array contains spike times in sample units.
    amplitude_vectors : ndarray of shape (num_clusters, recording_channels), dtype=float32
        Amplitude vector representation for each cluster, used for merging across segments.
    """

    def __init__(self, *, start_time, duration, spike_clusters, amplitude_vectors):
        """
        Initializes a SortedSegment object.

        Parameters
        ----------
        start_time : int
            Sample index where the segment begins in the global recording.
        duration : int
            Number of samples covered by this segment.
        spike_clusters : list of ndarray
            Spike time arrays, one per cluster, in sample units.
        amplitude_vectors : ndarray
            One amplitude vector per cluster, used for matching across segments.
        """
        
        self.start_time = start_time
        self.duration = duration
        self.spike_clusters = spike_clusters
        self.amplitude_vectors = amplitude_vectors
    
    def __len__(self):
        """
        Returns the number of clusters in the segment.

        Returns
        -------
        int
            Number of clusters.
        """
        return len(self.spike_clusters)


def detect_and_cluster_spikes(
    data, 
    *, 
    min_pos, 
    min_values, 
    probe_geometry, 
    mad_thresholds, 
    min_spikes_per_cluster, 
    merging_threshold
):
    """
    Detect and cluster spike waveforms from extracellular recordings.

    This function:
    - Selects channels by amplitude score and detects candidate spike peaks.
    - Extracts waveforms and performs hierarchical clustering.
    - Refines and verifies clusters using inter-channel features and waveform similarity.
    - Subtracts the mean waveform from the data to iteratively reveal overlapping spikes.
    - Updates detection buffers and continues until no further clusters can be isolated.

    Parameters
    ----------
    data : ndarray of shape (num_samples, recording_channels, samples_per_ms), dtype=float32
        Preprocessed extracellular recording array.
    min_pos : ndarray of shape (recording_channels, num_milliseconds), dtype=int
        Sample index of minimum value per ms per channel.
    min_values : ndarray of shape (recording_channels, num_milliseconds), dtype=float32
        Minimum value per ms per channel.
    probe_geometry : ndarray of shape (recording_channels, 2), dtype=float32
        2D layout of electrode sites.
    mad_thresholds : ndarray of shape (recording_channels,), dtype=float32
        MAD thresholds used for spike detection.
    min_spikes_per_cluster : float
        Minimum number of spikes required for a valid cluster.
    merging_threshold : float
        Similarity threshold (Euclidean) for accepting a split during clustering.

    Returns
    -------
    spike_clusters : list of ndarray
        List of 1D arrays, each containing spike positions for a detected neuron.
    amplitude_vectors : ndarray of shape (num_clusters, recording_channels), dtype=float32
        Amplitude fingerprint for each detected neuron (used in segment merging).
    """

    # Step 1: Score each channel by summed amplitude below MAD
    amplitude_scores = compute_amplitude_scores(min_values, mad_thresholds=mad_thresholds)

    # Initialize output containers
    spike_clusters = []
    amplitude_vectors = np.empty((0, data.shape[1]), dtype=np.float32)
    failed_channels = []

    # Step 2: Iteratively isolate spikes from one channel at a time
    while len(failed_channels) < data.shape[1]:
        # Mask already-processed channels by setting them to max + 1
        amplitude_scores[failed_channels] = amplitude_scores.max() + 1

        # Select the best candidate channel
        target_channel = np.argmin(amplitude_scores)

        # Detect candidate spike times on the selected channel
        spike_positions = detect_spike_peaks(
            min_pos = min_pos[target_channel], 
            min_values = min_values[target_channel],
            samples_per_ms = data.shape[2], 
            mad_threshold = mad_thresholds[target_channel]
        )

        # Reject if not enough spikes are detected
        if len(spike_positions) < min_spikes_per_cluster:
            failed_channels.append(target_channel)
            continue

        # Step 3: Select at most 5 nearest channels to target electrode
        closest_channels = np.argsort(
            np.linalg.norm(probe_geometry - probe_geometry[target_channel], axis=1)
        )[:5]

        # Extract waveforms from those channels around each spike
        dataset = extract_waveforms(
            data, 
            spike_positions = spike_positions, 
            channels = closest_channels
        )

        # Step 4: Hierarchically isolate the dominant neuron
        spike_positions, mean_waveform = perform_hierarchical_clustering(
            dataset, 
            spike_positions = spike_positions, 
            mean_waveform = dataset.mean(0),
            samples_per_ms = data.shape[2], 
            min_spikes_per_cluster = min_spikes_per_cluster, 
            merging_threshold = merging_threshold,
            num_channels = closest_channels.shape[0]
        )

        # Reject if clustering failed
        if len(spike_positions) < min_spikes_per_cluster:
            failed_channels.append(target_channel)
            continue

        # Step 5: Get mean spike waveform and compute reference template
        spike_template = mean_waveform.reshape(closest_channels.shape[0], -1)
        reference_vector = compute_inter_channel_differences(spike_template)

        # Re-detect all local minima (no threshold) to refine spike candidates
        spike_positions = detect_spike_peaks(
            min_pos = min_pos[target_channel], 
            min_values = min_values[target_channel], 
            samples_per_ms = data.shape[2], 
            mad_threshold = 0.0
        )

        # Extract features from full channel set (in ms space)
        features = extract_spike_features(
            min_values, 
            spike_positions = spike_positions // data.shape[2], 
            channels = closest_channels
        )
        
        # Filter out spikes with poor projection onto the template
        template_features = spike_template.min(1)
        threshold = template_features @ template_features / 2
        spike_positions = spike_positions[features @ template_features > threshold]

        # Step 6: Extract filtered dataset
        dataset = extract_waveforms(
            data, 
            spike_positions = spike_positions, 
            channels = closest_channels
        )

        # Step 7: Perform final clustering
        spike_positions, mean_waveform = perform_final_clustering(
            dataset, 
            spike_positions = spike_positions, 
            mean_waveform = dataset.mean(0),
            reference_vector = reference_vector, 
            min_spikes_per_cluster = min_spikes_per_cluster, 
            merging_threshold = merging_threshold,
            num_channels = closest_channels.shape[0]
        )

        # Final rejection: low amplitude or low count
        if len(spike_positions) < min_spikes_per_cluster or (
            mean_waveform[data.shape[2]] > mad_thresholds[target_channel]
        ):
            failed_channels.append(target_channel)
            continue

        # Step 8: Extract final amplitude vector
        negative_amplitudes = compute_average_spike_amplitude(
            min_values, 
            spike_positions = spike_positions // data.shape[2]
        )

        # Identify channels with strong signal for subtraction
        subtraction_channels = np.sort(
            np.where(negative_amplitudes < mad_thresholds / 2)[0]
        )

        # Subtract the mean waveform from affected channels
        subtract_average_waveform(
            data, 
            spike_positions = spike_positions, 
            channels = subtraction_channels
        )

        # Step 9: Save the detected neuron
        spike_clusters.append(spike_positions)
        amplitude_vectors = np.vstack([amplitude_vectors, -negative_amplitudes])

        # Step 10: Update detection statistics post-subtraction
        for peak in spike_positions // data.shape[2]:
            update_spike_statistics(
                data, 
                min_pos = min_pos, 
                min_values = min_values, 
                spike_position = peak, 
                channels = subtraction_channels, 
                amplitude_scores = amplitude_scores, 
                mad_thresholds = mad_thresholds
            )

    return spike_clusters, amplitude_vectors


class CyclicBufferContext:
    """
    Manages a cyclic buffer for memory-efficient spike sorting.

    This class handles:
    - Cyclic buffer allocation and wrapping.
    - Cache-efficient layout and minimum-value tracking.
    - Adaptive segmentation using a drift heuristic.

    Attributes
    ----------
    recording : Recording
        Input recording object (includes metadata and access logic).
    min_segment_length : int
        Minimum allowed segment duration (in milliseconds).
    buffer_size : int
        Total size of the cyclic buffer (4 * min_segment_length).
    samples_per_ms : int
        Number of samples per millisecond.
    recording_channels : int
        Number of recording channels.
    total_ms : int
        Duration of the recording in milliseconds.
    filtered : ndarray
        Raw bandpass-filtered signal, shape (samples, channels).
    filtered_cache : ndarray
        Reshaped view of filtered data for cache-efficient access.
    min_pos : ndarray
        Index of the minimum value per ms per channel.
    min_values : ndarray
        Minimum value per ms per channel.
    narrow, wide : int
        Filter parameters for DoG bandpass filtering.
    buffer_idx : int
        Current position in the cyclic buffer (millisecond index).
    current_ms : int
        Global time of the next segment to process.
    next_ms : int
        Global time of the next ms to load into the buffer.
    mad_thresholds : ndarray
        Channel-wise detection threshold based on MAD.
    """

    def __init__(self, *, recording, min_segment_length):
        """
        Initialize the cyclic buffer and configure tracking variables.

        Parameters
        ----------
        recording : Recording
            Recording object with geometry and data access methods.
        min_segment_length : int
            Minimum duration (in ms) of each valid segment.
        """

        self.recording = recording
        self.min_segment_length = min_segment_length
        self.buffer_size = min_segment_length * 4
        self.samples_per_ms = recording.samples_per_ms
        self.recording_channels = recording.recording_channels
        self.total_ms = recording.num_samples // self.samples_per_ms - 1

        # Allocate cyclic buffers
        self.filtered = np.empty((self.buffer_size * self.samples_per_ms, self.recording_channels), dtype=np.float32)
        self.filtered_cache = self.filtered.reshape(self.buffer_size, self.recording_channels, self.samples_per_ms)
        self.min_pos = np.empty((self.buffer_size, self.recording_channels), dtype=np.intp)
        self.min_values = np.empty((self.buffer_size, self.recording_channels), dtype=np.float32)

        # Define filter windows (relative to sampling rate)
        self.narrow = int(self.samples_per_ms / 9.5)
        self.wide = int(self.samples_per_ms / 4.75)

        # Tracking positions in the circular buffer
        self.buffer_idx = 0  # Position in the cyclic buffer (wraps around)
        self.current_ms = 1  # Current processing timestamp in milliseconds
        self.next_ms = 1  # Next millisecond to load into the buffer

    def initialize_buffer(self, *, detection_sensitivity, detection_polarity):
        """
        Load initial data block and compute MAD thresholds for spike detection.

        Parameters
        ----------
        detection_sensitivity : float
            Scaling factor for MAD threshold (e.g., 10.0).
        detection_polarity : float
            Scalar applied to the signal prior to spike detection.
        """
        
        # Determine how much data can be loaded initially without exceeding total duration
        initial_load_size = self.samples_per_ms * min(
            self.buffer_size, 
            self.total_ms - self.current_ms
        )

        # Read raw data from the recording, ensuring adequate padding for the bandpass filter
        raw_data = self.recording.read(
            start = self.samples_per_ms - self.wide * 4,
            num_samples = initial_load_size + self.wide * 8 
        )

        # Apply bandpass filtering
        apply_bandpass_filter(
            raw_data = raw_data, 
            filtered_data = self.filtered[:initial_load_size], 
            narrow = self.narrow, 
            wide = self.wide, 
            detection_polarity = detection_polarity
        )

        # Compute per-channel detection thresholds
        self.mad_thresholds = -detection_sensitivity * np.median(
            np.abs(self.filtered[:initial_load_size:self.samples_per_ms]), 
            axis = 0
        )

        # Compute per-millisecond minimum values and positions for cache-efficient access
        transpose_and_compute_minima(
            self.filtered[:initial_load_size],
            min_pos = self.min_pos[:initial_load_size // self.samples_per_ms],
            min_values = self.min_values[:initial_load_size // self.samples_per_ms],
            samples_per_ms = self.samples_per_ms
        )

        # Update tracking variable for the next millisecond to be processed
        self.next_ms += initial_load_size // self.samples_per_ms

    def load_next_batch(self, *, detection_polarity):
        """
        Load and filter the next chunk of data into the cyclic buffer.

        Parameters
        ----------
        detection_polarity : float
            Scalar applied to the signal prior to spike detection.
        """

        while self.next_ms < self.total_ms and (self.next_ms - self.current_ms) < self.buffer_size:
            # Compute the start position in the cyclic buffer (handling wraparound)
            start = self.samples_per_ms * (
                (self.buffer_idx + self.next_ms - self.current_ms) % self.buffer_size
            )

            # Determine how many samples to load
            if start < self.buffer_idx * self.samples_per_ms:
                end = self.samples_per_ms * min(
                    self.buffer_idx, 
                    self.buffer_idx + self.total_ms - self.current_ms - self.buffer_size
                )
            else:
                end = self.samples_per_ms * min(
                    self.buffer_size, 
                    self.buffer_idx + self.total_ms - self.current_ms
                )

            # Read raw data from the recording file with extra padding for filtering
            raw_data = self.recording.read(
                start = self.next_ms * self.samples_per_ms - self.wide * 4,
                num_samples = end - start + self.wide * 8 
            )

            # Apply bandpass filter and store in cyclic buffer
            apply_bandpass_filter(
                raw_data = raw_data, 
                filtered_data = self.filtered[start:end], 
                narrow = self.narrow, 
                wide = self.wide, 
                detection_polarity = detection_polarity
            )

            # Compute per-millisecond min values and positions for cache-efficient processing
            transpose_and_compute_minima(
                self.filtered[start:end],
                min_pos = self.min_pos[start // self.samples_per_ms:end // self.samples_per_ms],
                min_values = self.min_values[start // self.samples_per_ms:end // self.samples_per_ms],
                samples_per_ms = self.samples_per_ms
            )

            # Advance next_ms by the number of milliseconds processed
            self.next_ms += (end - start) // self.samples_per_ms

    
    def determine_segment_duration(self):
        """
        Determine the optimal segment duration using drift-based heuristics.

        Returns
        -------
        int
            Chosen segment duration in milliseconds.
        """
        # If less than 2 * min_segment_length data is available, return what remains
        if self.next_ms - self.current_ms < self.min_segment_length * 2:
            return self.next_ms - self.current_ms

        # Calculate the end position in the cyclic buffer, considering wraparound
        buffer_end = self.buffer_idx + (
            self.next_ms - self.current_ms
        )

        # Copy the relevant section of min_values to avoid in-place mutation
        if buffer_end < self.buffer_size:
            segment_data = self.min_values[self.buffer_idx:buffer_end].copy()
        else:
            segment_data = np.concatenate([
                self.min_values[self.buffer_idx:].copy(),
                self.min_values[:buffer_end - self.buffer_size].copy()
            ])

        # Compute drift heuristic values to determine the best segment boundary
        drift_values = compute_drift_heuristic(
            segment_data, 
            mad_thresholds = self.mad_thresholds, 
            min_segment_length = self.min_segment_length
        )

        # Find the index where the drift heuristic suggests a segment boundary
        split_idx = np.argmax(drift_values)

        # Ensure that the segment cannot be split further
        if split_idx >= self.min_segment_length:
            split_idx = np.argmax(
                drift_values[:split_idx - self.min_segment_length + 1]
            )

        return split_idx + self.min_segment_length


    def extract_segment(self, *, segment_duration):
        """
        Extract a filtered segment from the buffer and handle wraparound.

        Parameters
        ----------
        segment_duration : int
            Number of milliseconds to extract.

        Returns
        -------
        tuple
            - filtered_segment : (ms, channels, samples_per_ms)
            - min_pos : (channels, ms)
            - min_values : (channels, ms)
        """
        # Compute start and end indices in the cyclic buffer
        start = self.buffer_idx
        end = (self.buffer_idx + segment_duration) % self.buffer_size

        # Update tracking variables
        self.buffer_idx = end
        self.current_ms += segment_duration

        # Case 1: No wraparound -> Extract directly
        if end > start:
            return (
                self.filtered_cache[start:end],
                self.min_pos[start:end].T.copy(),
                self.min_values[start:end].T.copy()
            )

        # Case 2: Wraparound -> Concatenate segments from both buffer ends
        return (
            np.concatenate([
                self.filtered_cache[start:], 
                self.filtered_cache[:end]
            ]),
            np.concatenate([
                self.min_pos[start:], 
                self.min_pos[:end]
            ]).T.copy(),
            np.concatenate([
                self.min_values[start:], 
                self.min_values[:end]
            ]).T.copy()
        )

def segment_and_sort_spikes(
    recording, 
    *, 
    min_segment_length, 
    detection_sensitivity, 
    min_spikes_per_cluster,
    merging_threshold , 
    detection_polarity, 
    verbose
):
    """
    Performs adaptive spike sorting on a continuous extracellular recording.

    This function:
    - Segments the recording based on drift using a cyclic buffer.
    - Detects and clusters spikes independently within each segment.
    - Returns a list of SortedSegment objects.

    Parameters
    ----------
    recording : Recording
        Input object providing raw signal data and probe metadata.
    min_segment_length : float, optional
        Minimum segment duration in milliseconds (must be >= 100).
    detection_sensitivity : float, optional
        Scaling factor for spike detection threshold (based on MAD).
    min_spikes_per_cluster : float, optional
        Minimum number of spikes required to retain a cluster (must be >= 2).
    merging_threshold : float, optional
        Inter-cluster similarity threshold for merging (0 < value < 1).
    detection_polarity : flaot, optional
        Scalar applied to the signal prior to spike detection.
    verbose : bool, optional
        If True, display progress bar during processing.

    Returns
    -------
    list of SortedSegment
        One entry per processed segment, containing clustered spikes and amplitude vectors.
    """

    # Initialize cyclic buffer context for real-time streaming from disk
    ctx = CyclicBufferContext(
        recording = recording, 
        min_segment_length = min_segment_length
    )

    # Load initial segment and compute channel thresholds
    ctx.initialize_buffer(
        detection_sensitivity = detection_sensitivity, 
        detection_polarity = detection_polarity
    )

    # Initialize progress display
    progress_bar = tqdm(
        total = ctx.total_ms - ctx.current_ms,
        desc = "SpikeSift",
        unit = "ms of data",
        ncols = 100,
        disable = not verbose
    )

    # Final output list of SortedSegment objects
    sorted_segments = [] 
    
    # Loop over the entire recording
    while ctx.current_ms < ctx.total_ms:
        # Compute segment start time (in samples)
        segment_start = recording.recording_offset + (
            ctx.current_ms * ctx.samples_per_ms
        )
    
        # Use drift-aware heuristic to determine segment duration
        segment_duration = ctx.determine_segment_duration()

        # Extract filtered data and minimum values for this segment
        current_segment, argmin, minval = ctx.extract_segment(
            segment_duration = segment_duration
        )

        # Detect and cluster spikes in this segment
        spike_clusters, amplitude_vectors = detect_and_cluster_spikes(
            current_segment, 
            min_pos = argmin, 
            min_values = minval,
            probe_geometry = recording.probe_geometry, 
            mad_thresholds = ctx.mad_thresholds, 
            min_spikes_per_cluster = min_spikes_per_cluster, 
            merging_threshold = merging_threshold
        )

        # Offset timestamps to global time (in samples)
        spike_clusters = [p + segment_start for p in spike_clusters]

        # Save the result as a SortedSegment
        sorted_segments.append(
            SortedSegment(
                start_time = segment_start,
                duration = segment_duration * ctx.samples_per_ms,
                spike_clusters = spike_clusters,
                amplitude_vectors = amplitude_vectors
            )
        )

        # Update progress and load more data into the buffer
        progress_bar.update(segment_duration)
        ctx.load_next_batch(
            detection_polarity = detection_polarity
        )

    progress_bar.close()

    # Extend the first segment backwards to reach the start of the recording
    sorted_segments[0].duration = (
        sorted_segments[0].start_time + sorted_segments[0].duration
        - recording.recording_offset
    )
    sorted_segments[0].start_time = recording.recording_offset
    
    # Extend the last segment to reach the true end of the recording
    sorted_segments[-1].duration = (
        recording.recording_offset + recording.num_samples 
        - sorted_segments[-1].start_time
    )
    return sorted_segments

def find_optimal_assignment(
    amplitude_vectors1, 
    amplitude_vectors2, 
    *, 
    probe_geometry, 
    max_drift
):
    """
    Find the optimal alignment between clusters from two adjacent segments.

    This function:
    - Simulates vertical drift in opposite directions on each segment.
    - Computes all-to-all Euclidean distances between adjusted amplitude vectors.
    - Uses the Hungarian algorithm to solve the minimum-cost assignment problem.
    - Returns the assignment that minimizes total cost across all tested shifts.

    Parameters
    ----------
    amplitude_vectors1 : ndarray of shape (num_units, recording_channels), dtype=float32
        Amplitude-based representations of clusters in the first segment.
    amplitude_vectors2 : ndarray of shape (num_units, recording_channels), dtype=float32
        Amplitude-based representations of clusters in the second segment.
    probe_geometry : ndarray of shape (recording_channels, 2), dtype=float32
        2D (x, y) coordinates for each electrode.
    max_drift : float
        Maximum tested vertical shift (in micrometers) between segments.

    Returns
    -------
    optimal_assignment : ndarray of shape (num_units,), dtype=intp
        Mapping from clusters in the first segment to clusters in the second.
        Clusters with no valid match are assigned -1.
    """

    # Stores the best cluster assignment
    optimal_assignment = None 
    optimal_cost = np.inf

    # Search over a range of symmetric drift displacements
    for drift_step in np.arange(-max_drift, max_drift + 1, 5):
        # Apply vertical drift: one segment up, the other down
        shifted_v1 = simulate_drift(
            amplitude_vectors1, 
            probe_geometry = probe_geometry, 
            step = drift_step / 2
        )
        shifted_v2 = simulate_drift(
            amplitude_vectors2, 
            probe_geometry = probe_geometry, 
            step = -drift_step / 2
        )

        # Compute pairwise cost matrix (Euclidean distances)
        cost_matrix = cdist(
            shifted_v1, 
            shifted_v2, 
            metric = 'euclidean'
        )

        # Solve optimal one-to-one assignment via Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Compute total assignment cost
        total_cost = cost_matrix[row_indices, col_indices].sum()

        # Update best assignment if lower cost is found
        if total_cost < optimal_cost:
            optimal_cost = total_cost
            optimal_assignment = -np.ones(cost_matrix.shape[0], dtype = 'intp')
            optimal_assignment[row_indices] = col_indices

    return optimal_assignment