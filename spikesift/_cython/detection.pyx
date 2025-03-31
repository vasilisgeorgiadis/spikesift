import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def compute_amplitude_scores(
    cnp.ndarray[cnp.float32_t, ndim = 2] min_values, 
    *, 
    cnp.ndarray[cnp.float32_t, ndim = 1] mad_thresholds
):
    """
    Compute the summed amplitude deviation for each channel relative to its MAD threshold.

    This function:
    - Iterates over a matrix of per-channel minimum voltage values over time.
    - Computes the negative deviation from the MAD threshold (if any).
    - Aggregates the sum of deviations per channel, producing a simple scalar score
      for how strongly each channel shows sub-threshold activity.

    Used to guide spike detection or to score per-channel signal strength.

    Parameters
    ----------
    min_values : ndarray of shape (recording_channels, num_samples), dtype=float32
        Input matrix of minimum voltage values per channel over time.
    mad_thresholds : ndarray of shape (recording_channels,), dtype=float32
        Per-channel noise thresholds (Median Absolute Deviations, or MADs).

    Returns
    -------
    amplitude_scores : ndarray of shape (recording_channels,), dtype=float32
        Total summed deviation below the MAD threshold for each channel.
        Higher magnitude implies stronger deflections from noise floor.
    """

    # Extract input dimensions
    cdef cnp.intp_t recording_channels = min_values.shape[0]
    cdef cnp.intp_t num_samples = min_values.shape[1]

    # Get pointer to input signal for fast access
    cdef cnp.float32_t *min_values_ptr = &min_values[0, 0]

    # Allocate output array for per-channel scores
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] amplitude_scores = np.zeros(recording_channels, dtype = 'float32')
    cdef cnp.float32_t *amplitude_scores_ptr = &amplitude_scores[0]

    # Loop variables
    cdef cnp.intp_t channel_idx
    cdef cnp.intp_t sample_idx = 0

    # Iterate over each channel and accumulate thresholded deviations
    for channel_idx in range(recording_channels):
        for _ in range(num_samples):
            amplitude_scores_ptr[channel_idx] += min(0.0,
                min_values_ptr[sample_idx] - mad_thresholds[channel_idx]
            )
            sample_idx += 1

    return amplitude_scores

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def detect_spike_peaks(
    *, 
    cnp.ndarray[cnp.intp_t, ndim = 1] min_pos, 
    cnp.ndarray[cnp.float32_t, ndim = 1] min_values, 
    cnp.intp_t samples_per_ms, 
    cnp.float32_t mad_threshold
):
    """
    Detect spike peaks based on local minima in per-millisecond voltage values.

    This function:
    - Identifies local minima whose amplitude is below a specified MAD threshold.
    - Enforces temporal locality by ensuring that a candidate minimum:
        - Is not preceded by a smaller or equal value (leftward check).
        - Is not followed by a smaller value (rightward check).
    - Converts relative within-millisecond minima to absolute sample indices.

    Used to identify candidate spike times before waveform extraction.

    Parameters
    ----------
    min_pos : ndarray of shape (num_milliseconds,), dtype=intp
        Relative sample index of the minimum value within each millisecond.
    min_values : ndarray of shape (num_milliseconds,), dtype=float32
        Minimum voltage value per millisecond.
    samples_per_ms : int
        Number of samples per millisecond.
    mad_threshold : float
        Threshold for spike detection. Must be negative for negative-going spikes.

    Returns
    -------
    spike_indices : ndarray of shape (num_detected_peaks,), dtype=intp
        Absolute sample indices of detected spike peaks.
    """

    # Get total number of time blocks
    cdef cnp.intp_t num_milliseconds = min_pos.shape[0]

    # Allocate space for all possible spikes (max possible = num_milliseconds)
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] spike_indices = np.zeros(num_milliseconds, dtype = 'intp')
    cdef cnp.intp_t *spike_indices_ptr = &spike_indices[0]

    # Declare loop variables
    cdef cnp.intp_t ms_idx             # Millisecond index
    cdef cnp.intp_t peak_count = 0     # Count of detected spikes

    # Avoid boundaries: leave 2ms on each side for waveform extraction
    for ms_idx in range(2, num_milliseconds - 2):
        if (
            min_values[ms_idx] < mad_threshold  # Amplitude must be below the threshold
            and (min_pos[ms_idx - 1] <= min_pos[ms_idx] or min_values[ms_idx - 1] > min_values[ms_idx])  # Left check
            and (min_pos[ms_idx] <= min_pos[ms_idx + 1] or min_values[ms_idx] <= min_values[ms_idx + 1])  # Right check
        ):
            # Convert to absolute sample index (millisecond offset + within-ms offset)
            spike_indices_ptr[peak_count] = min_pos[ms_idx] + ms_idx * samples_per_ms
            peak_count += 1

    # Return slice containing only valid spikes
    return spike_indices[:peak_count]

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def extract_spike_features(
    cnp.ndarray[cnp.float32_t, ndim = 2] min_values, 
    *, 
    cnp.ndarray[cnp.intp_t, ndim = 1] spike_positions, 
    cnp.ndarray[cnp.intp_t, ndim = 1] channels
):
    """
    Extract feature vectors for detected spike events using per-channel minimum voltages.

    This function:
    - Selects a subset of channels for feature extraction.
    - Gathers the minimum voltage values at spike positions for each selected channel.
    - Produces a matrix of shape (num_spikes, num_channels) suitable for a fast comparison.

    Parameters
    ----------
    min_values : ndarray of shape (num_channels, num_milliseconds), dtype=float32
        Matrix of minimum voltage values per channel and millisecond.
    spike_positions : ndarray of shape (num_spikes,), dtype=intp
        Indices (in milliseconds) where spike peaks were detected.
    channels : ndarray of shape (num_channels,), dtype=intp
        Indices of channels from which to extract features (i.e., spatial neighborhood).

    Returns
    -------
    spike_features : ndarray of shape (num_spikes, num_channels), dtype=float32
        Feature matrix where each row corresponds to a spike event and
        each column corresponds to a selected channel's value.
    """

    # Extract dimensions
    cdef cnp.intp_t num_spikes = spike_positions.shape[0]
    cdef cnp.intp_t num_channels = channels.shape[0]

    # Allocate output feature matrix
    cdef cnp.ndarray[cnp.float32_t, ndim = 2] spike_features = np.empty((num_spikes, num_channels), dtype = 'float32')
    cdef cnp.float32_t *spike_features_ptr = &spike_features[0,0]

    # Declare loop variables
    cdef cnp.intp_t spike_idx           # Index over spike events
    cdef cnp.intp_t channel_offset      # Index over selected channels
    cdef cnp.intp_t channel_idx         # Physical channel index
    cdef cnp.intp_t feature_idx = 0     # Index of spike_features array

    # Extract features: iterate over each spike and each selected channel
    for spike_idx in range(num_spikes):
        for channel_offset in range(num_channels):
            channel_idx = channels[channel_offset]
            spike_features_ptr[feature_idx] = min_values[channel_idx, spike_positions[spike_idx]]
            feature_idx += 1
            
    return spike_features

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def extract_waveforms(
    cnp.ndarray[cnp.float32_t, ndim = 3] data, 
    *, 
    cnp.ndarray[cnp.intp_t, ndim = 1] spike_positions, 
    cnp.ndarray[cnp.intp_t, ndim = 1] channels
):
    """
    Extract multi-channel waveform snippets around detected spike times.

    This function:
    - Extracts spike-centered waveforms from a 3D extracellular recording.
    - Uses a fixed 2 ms window: 1 ms before and 1 ms after each spike.
    - Gathers waveforms from a fixed set of spatially nearby channels.
    - Returns a flattened matrix suitable for clustering and visualization.

    All memory access is performed using precomputed offsets and raw pointers
    for cache efficiency and maximum throughput.

    Parameters
    ----------
    data : ndarray of shape (num_samples, recording_channels, samples_per_ms), dtype=float32
        Extracellular recording after DoG filtering and transposition.
    spike_positions : ndarray of shape (num_spikes,), dtype=intp
        Absolute spike times, in sample units (not milliseconds).
    channels : ndarray of shape (num_channels,), dtype=intp
        Channel indices used for waveform extraction (e.g., local neighborhood).

    Returns
    -------
    spike_waveforms : ndarray of shape (num_spikes, 2 * num_channels * samples_per_ms), dtype=float32
        Flattened spike-aligned waveform snippets for each event.
        Each row corresponds to one spike, concatenating selected channels over time.
    """

    # Extract dimensions
    cdef cnp.intp_t num_spikes = spike_positions.shape[0]
    cdef cnp.intp_t num_channels = channels.shape[0]
    cdef cnp.intp_t recording_channels = data.shape[1]
    cdef cnp.intp_t samples_per_ms = data.shape[2]

    # Compute flattened memory block sizes
    cdef cnp.intp_t block_size = recording_channels * samples_per_ms  
    cdef cnp.intp_t waveform_length = 2 * num_channels * samples_per_ms
    
    # Raw pointer to input data for fast memory access
    cdef cnp.float32_t *data_ptr = &data[0,0,0]

    # Allocate output waveform matrix
    cdef cnp.ndarray[cnp.float32_t, ndim = 2] spike_waveforms = np.zeros((num_spikes, waveform_length), dtype = 'float32')
    cdef cnp.float32_t *spike_waveforms_ptr = &spike_waveforms[0,0]

    # Declare loop variables
    cdef cnp.intp_t spike_idx          # Index over spikes
    cdef cnp.intp_t channel_offset     # Index over selected channels
    cdef cnp.intp_t channel_idx        # Absolute channel index
    cdef cnp.intp_t sample_idx         # Index over samples within 1 ms
    cdef cnp.intp_t ms_idx             # Whole millisecond index
    cdef cnp.intp_t offset             # Sub-ms sample index within millisecond
    cdef cnp.intp_t data_idx           # Linearized index into data_ptr
    cdef cnp.intp_t waveform_idx = 0   # Flat write position in output matrix

    # Loop through each detected spike
    for spike_idx in range(num_spikes):
        # Compute ms block and sample offset within that block
        ms_idx = spike_positions[spike_idx] // samples_per_ms
        offset = spike_positions[spike_idx] % samples_per_ms

        # Loop through selected channels (e.g., spatial neighbors)
        for channel_offset in range(num_channels):
            channel_idx = channels[channel_offset]

            # Compute base offset to current (ms, channel)
            data_idx = samples_per_ms * (
                ms_idx * recording_channels + channel_idx
            )

            # 1. Pre-spike: trailing samples of -1ms
            for sample_idx in range(offset, samples_per_ms):
                spike_waveforms_ptr[waveform_idx] = data_ptr[data_idx + sample_idx - block_size]
                waveform_idx += 1

            # 2. Spike-centered: 0ms
            for sample_idx in range(samples_per_ms):
                spike_waveforms_ptr[waveform_idx] = data_ptr[data_idx + sample_idx]
                waveform_idx += 1

            # 3. Post-spike: leading samples of +1ms
            for sample_idx in range(offset):
                spike_waveforms_ptr[waveform_idx] = data_ptr[data_idx + sample_idx + block_size]
                waveform_idx += 1

    return spike_waveforms
