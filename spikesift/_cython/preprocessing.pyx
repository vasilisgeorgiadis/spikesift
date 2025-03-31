import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

ctypedef fused DTYPE_T:
    cnp.int8_t
    cnp.uint8_t
    cnp.int16_t
    cnp.uint16_t
    cnp.int32_t
    cnp.uint32_t
    cnp.int64_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def apply_bandpass_filter(
    *, 
    cnp.ndarray[DTYPE_T, ndim = 2] raw_data, 
    cnp.ndarray[cnp.float32_t, ndim = 2] filtered_data, 
    cnp.intp_t narrow, 
    cnp.intp_t wide, 
    cnp.float32_t detection_polarity
):
    """
    Apply a Difference-of-Gaussians (DoG) bandpass filter to extracellular voltage recordings.

    This function applies two approximated Gaussian filters using four recursive box filters,
    and subtracts them to produce a bandpass effect. It supports polarity inversion for
    detecting both positive and negative spikes. The implementation is fully in-place,
    memory-efficient, and optimized for cache performance and CPU throughput.

    Parameters
    ----------
    raw_data : ndarray of shape (num_samples + 8 * wide, recording_channels), dtype=DTYPE_T
        Input voltage signal. Must include 4 * wide extra samples on both sides to prevent boundary artifacts.
    filtered_data : ndarray of shape (num_samples, recording_channels), dtype=float32
        Output buffer to store the bandpass-filtered signal. Modified in-place.
    narrow : int
        Width parameter (in samples) for the narrow Gaussian approximation.
    wide : int
        Width parameter (in samples) for the wide Gaussian approximation.
    detection_polarity : float
        Scalar applied to the signal prior to spike detection.

    Returns
    -------
    None
        The filtered output is written directly to filtered_data.
    """
    
    # Extract dimensions
    cdef cnp.intp_t num_samples = raw_data.shape[0]
    cdef cnp.intp_t recording_channels = raw_data.shape[1]

    # Convert spatial filter widths to flattened buffer offsets
    cdef cnp.intp_t narrow_offset = 4 * narrow * recording_channels
    cdef cnp.intp_t wide_offset = 4 * wide * recording_channels

    # Allocate circular buffer sizes as powers of 2 for efficient modular indexing
    cdef cnp.intp_t narrow_buffer_size = 2 ** int(
        np.ceil(np.log2(2 * narrow_offset + 4 * recording_channels))
    )
    cdef cnp.intp_t wide_buffer_size = 2 ** int(
        np.ceil(np.log2(2 * wide_offset + 4 * recording_channels))
    )

    # Create bitmasks for fast circular indexing (avoids modulus operator)
    cdef cnp.intp_t narrow_mask = narrow_buffer_size - 1
    cdef cnp.intp_t wide_mask = wide_buffer_size - 1

    # Initialize circular buffer indices
    cdef cnp.intp_t narrow_prev_idx = (-2 * narrow_offset) & narrow_mask
    cdef cnp.intp_t narrow_curr_idx = 0
    cdef cnp.intp_t narrow_next_idx = (4 * recording_channels) & narrow_mask
    cdef cnp.intp_t wide_prev_idx = (-2 * wide_offset) & wide_mask
    cdef cnp.intp_t wide_curr_idx = 0
    cdef cnp.intp_t wide_next_idx = (4 * recording_channels) & wide_mask

    # Raw memory pointers for efficient access
    cdef DTYPE_T *raw_ptr = &raw_data[0, 0]
    cdef cnp.float32_t *filtered_ptr = &filtered_data[0, 0]

    # Allocate circular filter buffers and expose raw pointers
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] narrow_buffer = np.zeros(narrow_buffer_size, dtype = 'float32')
    cdef cnp.float32_t *narrow_buffer_ptr = &narrow_buffer[0]
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] wide_buffer = np.zeros(wide_buffer_size, dtype = 'float32')
    cdef cnp.float32_t *wide_buffer_ptr = &wide_buffer[0]

    # Precompute normalization constants for recursive filters
    cdef cnp.float32_t narrow_weight = -detection_polarity / (2 * narrow + 1) ** 4
    cdef cnp.float32_t wide_weight = detection_polarity / (2 * wide + 1) ** 4

    # Declare loop variable
    cdef cnp.intp_t sample_idx

    # Zero-initialize the first filtered sample per channel
    for sample_idx in range(recording_channels):
        filtered_ptr[sample_idx] = 0

    # Phase 1: Warm-up pass to fill recursive buffers with initial state
    for sample_idx in range(recording_channels + wide_offset * 2):
        # Advance circular indices
        narrow_prev_idx = (narrow_prev_idx + 4) & narrow_mask
        narrow_curr_idx = (narrow_curr_idx + 4) & narrow_mask
        narrow_next_idx = (narrow_next_idx + 4) & narrow_mask
        wide_prev_idx = (wide_prev_idx + 4) & wide_mask
        wide_curr_idx = (wide_curr_idx + 4) & wide_mask
        wide_next_idx = (wide_next_idx + 4) & wide_mask

        # Insert next raw sample into buffer (zero-padded on left edge)
        narrow_buffer_ptr[narrow_next_idx] = (
            raw_ptr[sample_idx - wide_offset + narrow_offset] 
            if sample_idx - wide_offset + narrow_offset >= 0 else 0
        )
        wide_buffer_ptr[wide_next_idx] = raw_ptr[sample_idx]

        # Apply three recursive filter stages (final output after 4th stage)
        narrow_buffer_ptr[narrow_next_idx + 1] = narrow_buffer_ptr[narrow_curr_idx + 1] + (
            narrow_buffer_ptr[narrow_next_idx] - narrow_buffer_ptr[narrow_prev_idx]
        )
        narrow_buffer_ptr[narrow_next_idx + 2] = narrow_buffer_ptr[narrow_curr_idx + 2] + (
            narrow_buffer_ptr[narrow_next_idx + 1] - narrow_buffer_ptr[narrow_prev_idx + 1]
        )
        narrow_buffer_ptr[narrow_next_idx + 3] = narrow_buffer_ptr[narrow_curr_idx + 3] + (
            narrow_buffer_ptr[narrow_next_idx + 2] - narrow_buffer_ptr[narrow_prev_idx + 2]
        )
    
        wide_buffer_ptr[wide_next_idx + 1] = wide_buffer_ptr[wide_curr_idx + 1] + (
            wide_buffer_ptr[wide_next_idx] - wide_buffer_ptr[wide_prev_idx]
        )
        wide_buffer_ptr[wide_next_idx + 2] = wide_buffer_ptr[wide_curr_idx + 2] + (
            wide_buffer_ptr[wide_next_idx + 1] - wide_buffer_ptr[wide_prev_idx + 1]
        )
        wide_buffer_ptr[wide_next_idx + 3] = wide_buffer_ptr[wide_curr_idx + 3] + (
            wide_buffer_ptr[wide_next_idx + 2] - wide_buffer_ptr[wide_prev_idx + 2]
        )

        # Accumulate filtered result into first row (per-channel buffer)
        filtered_ptr[sample_idx % recording_channels] += (
            (narrow_buffer_ptr[narrow_next_idx + 3] - narrow_buffer_ptr[narrow_prev_idx + 3]) * narrow_weight +
            (wide_buffer_ptr[wide_next_idx + 3] - wide_buffer_ptr[wide_prev_idx + 3]) * wide_weight
        )

    # Phase 2: Main loop for all remaining samples
    for sample_idx in range(recording_channels + wide_offset * 2, num_samples * recording_channels):
        # Advance circular indices
        narrow_prev_idx = (narrow_prev_idx + 4) & narrow_mask
        narrow_curr_idx = (narrow_curr_idx + 4) & narrow_mask
        narrow_next_idx = (narrow_next_idx + 4) & narrow_mask
        wide_prev_idx = (wide_prev_idx + 4) & wide_mask
        wide_curr_idx = (wide_curr_idx + 4) & wide_mask
        wide_next_idx = (wide_next_idx + 4) & wide_mask

        # Insert next raw sample
        narrow_buffer_ptr[narrow_next_idx] = raw_ptr[sample_idx - wide_offset + narrow_offset]
        wide_buffer_ptr[wide_next_idx] = raw_ptr[sample_idx]

        # Apply recursive filter stages
        narrow_buffer_ptr[narrow_next_idx + 1] = narrow_buffer_ptr[narrow_curr_idx + 1] + (
            narrow_buffer_ptr[narrow_next_idx] - narrow_buffer_ptr[narrow_prev_idx]
        )
        narrow_buffer_ptr[narrow_next_idx + 2] = narrow_buffer_ptr[narrow_curr_idx + 2] + (
            narrow_buffer_ptr[narrow_next_idx + 1] - narrow_buffer_ptr[narrow_prev_idx + 1]
        )
        narrow_buffer_ptr[narrow_next_idx + 3] = narrow_buffer_ptr[narrow_curr_idx + 3] + (
            narrow_buffer_ptr[narrow_next_idx + 2] - narrow_buffer_ptr[narrow_prev_idx + 2]
        )
    
        wide_buffer_ptr[wide_next_idx + 1] = wide_buffer_ptr[wide_curr_idx + 1] + (
            wide_buffer_ptr[wide_next_idx] - wide_buffer_ptr[wide_prev_idx]
        )
        wide_buffer_ptr[wide_next_idx + 2] = wide_buffer_ptr[wide_curr_idx + 2] + (
            wide_buffer_ptr[wide_next_idx + 1] - wide_buffer_ptr[wide_prev_idx + 1]
        )
        wide_buffer_ptr[wide_next_idx + 3] = wide_buffer_ptr[wide_curr_idx + 3] + (
            wide_buffer_ptr[wide_next_idx + 2] - wide_buffer_ptr[wide_prev_idx + 2]
        )

        # Compute and store final filtered output
        filtered_ptr[sample_idx - wide_offset * 2] = filtered_ptr[sample_idx - wide_offset * 2 - recording_channels] + (
            (narrow_buffer_ptr[narrow_next_idx + 3] - narrow_buffer_ptr[narrow_prev_idx + 3]) * narrow_weight +
            (wide_buffer_ptr[wide_next_idx + 3] - wide_buffer_ptr[wide_prev_idx + 3]) * wide_weight
        )

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def transpose_and_compute_minima(
    cnp.ndarray[cnp.float32_t, ndim = 2] data, 
    *, 
    cnp.ndarray[cnp.intp_t, ndim = 2] min_pos, 
    cnp.ndarray[cnp.float32_t, ndim = 2] min_values, 
    cnp.intp_t samples_per_ms
):
    """
    Restructure data for cache-efficient access and compute per-millisecond voltage minima.

    This function:
    - Rearranges a 2D input signal of shape (num_samples, recording_channels)
      into a 3D block layout (num_milliseconds, recording_channels, samples_per_ms),
      improving memory access patterns for subsequent computation.
    - Simultaneously computes the minimum value per millisecond for each channel,
      storing both the value and its within-millisecond sample position.

    All modifications are performed in-place using a temporary buffer for transposition.

    Parameters
    ----------
    data : ndarray of shape (num_samples, recording_channels), dtype=float32
        Input extracellular voltage signal. Modified in-place to optimize layout.
    min_pos : ndarray of shape (num_milliseconds, recording_channels), dtype=intp
        Output array storing the sample index of the minimum voltage within each millisecond.
    min_values : ndarray of shape (num_milliseconds, recording_channels), dtype=float32
        Output array storing the minimum voltage value per millisecond and channel.
    samples_per_ms : int
        Number of samples per millisecond (defines block size in time dimension).

    Returns
    -------
    None
        The input data is transposed in-place, and min_pos, min_values are updated.
    """
    
    # Extract dimensions
    cdef cnp.intp_t num_samples = data.shape[0]
    cdef cnp.intp_t recording_channels = data.shape[1]

    # Compute the size of a single transposed block: one ms worth of samples across all channels
    cdef cnp.intp_t block_size = recording_channels * samples_per_ms  

    # Raw pointer to the input data for fast access
    cdef cnp.float32_t *data_ptr = &data[0, 0]  

    # Allocate temporary buffer to hold the transposed block
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] temp_buffer = np.empty(block_size, dtype = 'float32')
    cdef cnp.float32_t *temp_buffer_ptr = &temp_buffer[0]

    # Declare loop variables
    cdef cnp.intp_t ms_idx           # millisecond index
    cdef cnp.intp_t channel_idx      # channel index
    cdef cnp.intp_t sample_idx       # sample index within millisecond
    cdef cnp.intp_t temp_idx         # index in temporary transposed buffer
    cdef cnp.intp_t data_idx         # index in flattened input array
    cdef cnp.intp_t min_position     # index of minimum value within current ms block
    cdef cnp.float32_t min_value     # value of the minimum

    # Iterate over each millisecond block
    for ms_idx in range(num_samples // samples_per_ms):
        for channel_idx in range(recording_channels):
            # Compute starting position of this channel in the current ms block
            temp_idx = channel_idx * samples_per_ms
            data_idx = ms_idx * block_size + channel_idx

            # Initialize minimum tracking with a safe high value
            min_value = data_ptr[data_idx] + 1
            min_position = 0

            # Transpose and search for minimum in a single pass
            for sample_idx in range(samples_per_ms):
                temp_buffer_ptr[temp_idx + sample_idx] = data_ptr[data_idx]

                # Track minimum value and its index
                if data_ptr[data_idx] < min_value:
                    min_position = sample_idx
                    min_value = data_ptr[data_idx]

                # Advance to the next sample on the same channel
                data_idx += recording_channels 

            # Store the results for this channel
            min_pos[ms_idx, channel_idx] = min_position
            min_values[ms_idx, channel_idx] = min_value

        # Copy transposed block back into data array (in-place overwrite)
        data_idx = ms_idx * block_size
        for sample_idx in range(block_size):
            data_ptr[data_idx + sample_idx] = temp_buffer_ptr[sample_idx]

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def compute_drift_heuristic(
    cnp.ndarray[cnp.float32_t, ndim = 2] min_values, 
    *, 
    cnp.ndarray[cnp.float32_t, ndim = 1] mad_thresholds, 
    cnp.intp_t min_segment_length
):
    """
    Compute a drift heuristic signal based on cumulative negative voltage deviations.

    This function:
    - Computes the cumulative sum of thresholded negative deviations per channel.
    - Overwrites min_values in-place with the cumulative values over time.
    - Calculates a second-order difference over a sliding window (2 * min_segment_length),
      producing a 1D signal that highlights regions of sustained drift.

    The resulting drift signal peaks at locations where strong voltage transitions
    suggest segment boundaries.

    Parameters
    ----------
    min_values : ndarray of shape (num_milliseconds, recording_channels), dtype=float32
        Input array of per-millisecond minimum voltages per channel.
        This array is modified in-place with cumulative deviation values.
    mad_thresholds : ndarray of shape (recording_channels,), dtype=float32
        Per-channel noise thresholds (e.g., MAD), used to suppress minor fluctuations.
    min_segment_length : int
        Minimum allowed segment duration in milliseconds.

    Returns
    -------
    drift_heuristic : ndarray of shape (num_milliseconds - 2 * min_segment_length + 1,), dtype=float32
        A 1D drift signal indicating where sustained deviations occur.
        Higher values suggest stronger drift and segment boundary candidates.
    """
    
    # Extract signal dimensions
    cdef cnp.intp_t num_milliseconds = min_values.shape[0]
    cdef cnp.intp_t recording_channels = min_values.shape[1]

    # Compute stride offsets and output size
    cdef cnp.intp_t block_size = min_segment_length * recording_channels  
    cdef cnp.intp_t num_values = num_milliseconds - 2 * min_segment_length + 1

    # Raw pointer to min_values for in-place modification
    cdef cnp.float32_t *min_values_ptr = &min_values[0, 0]

    # Output buffer for the 1D drift heuristic
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] drift_heuristic = np.zeros(num_values, dtype = 'float32')
    cdef cnp.float32_t *drift_heuristic_ptr = &drift_heuristic[0]

    # Running sum accumulator for each channel
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] cumulative_sums = np.zeros(recording_channels, dtype = 'float32')
    cdef cnp.float32_t *cumulative_sums_ptr = &cumulative_sums[0]

    # Declare loop indices
    cdef cnp.intp_t sample_idx = 0       # Flat index across all (ms x channels)
    cdef cnp.intp_t channel_idx          # Channel index
    cdef cnp.intp_t drift_idx            # Drift signal index (1D output)

    # Phase 1: Initialize cumulative sums for early ms range (no drift output yet)
    for _ in range(num_milliseconds - num_values):
        for channel_idx in range(recording_channels):
            cumulative_sums_ptr[channel_idx] += min(0.0,
                min_values_ptr[sample_idx] - mad_thresholds[channel_idx]
            )
            min_values_ptr[sample_idx] = cumulative_sums_ptr[channel_idx]
            sample_idx += 1

    # Phase 2: First drift point (second-order difference without left context)
    for channel_idx in range(recording_channels):
        cumulative_sums_ptr[channel_idx] += min(0.0,
            min_values_ptr[sample_idx] - mad_thresholds[channel_idx]
        )
        min_values_ptr[sample_idx] = cumulative_sums_ptr[channel_idx]
        drift_heuristic_ptr[0] += abs(
            min_values_ptr[sample_idx] - 
            2 * min_values_ptr[sample_idx - block_size]
        )
        sample_idx += 1

    # Phase 3: Main drift signal loop using second-order differencing
    for drift_idx in range(1, num_values):
        for channel_idx in range(recording_channels):
            cumulative_sums_ptr[channel_idx] += min(0.0,
                min_values_ptr[sample_idx] - mad_thresholds[channel_idx]
            )
            min_values_ptr[sample_idx] = cumulative_sums_ptr[channel_idx]
            drift_heuristic_ptr[drift_idx] += abs(
                min_values_ptr[sample_idx] - 
                2 * min_values_ptr[sample_idx - block_size] + 
                min_values_ptr[sample_idx - 2 * block_size]
            )
            sample_idx += 1

    return drift_heuristic
