import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative indexing for performance
def compute_average_spike_amplitude(
    cnp.ndarray[cnp.float32_t, ndim = 2] min_values, 
    *, 
    cnp.ndarray[cnp.intp_t, ndim = 1] spike_positions
):
    """
    Compute the average minimum voltage per channel over selected spike positions.

    This function:
    - Iterates over a list of spike time indices.
    - Accumulates the per-channel minimum voltages at those timepoints.
    - Returns the average minimum amplitude per channel.

    Used to summarize the amplitude profile of a spike cluster.

    Parameters
    ----------
    min_values : ndarray of shape (num_channels, num_milliseconds), dtype=float32
        Precomputed minimum voltage values for each channel over time.
    spike_positions : ndarray of shape (num_spikes,), dtype=intp
        Millisecond indices where spike peaks were detected.

    Returns
    -------
    avg_min_amplitude : ndarray of shape (num_channels,), dtype=float32
        Average minimum voltage value per channel across all selected spikes.
    """

    # Extract dimensions
    cdef cnp.intp_t num_spikes = spike_positions.shape[0]
    cdef cnp.intp_t num_channels = min_values.shape[0]

    # Allocate output array and get pointer
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] avg_min_amplitude = np.zeros(num_channels, dtype = 'float32')
    cdef cnp.float32_t *avg_min_amplitude_ptr = &avg_min_amplitude[0]

    # Declare loop variables
    cdef cnp.intp_t spike_idx
    cdef cnp.intp_t channel_idx

    # Accumulate minimum values across all spikes
    for spike_idx in range(num_spikes):  
        for channel_idx in range(num_channels):  
            avg_min_amplitude_ptr[channel_idx] += min_values[channel_idx, spike_positions[spike_idx]]

    # Normalize by the number of spikes
    for channel_idx in range(num_channels):
        avg_min_amplitude_ptr[channel_idx] /= num_spikes

    return avg_min_amplitude

@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative indexing for performance
def subtract_average_waveform(
    cnp.ndarray[cnp.float32_t, ndim = 3] data, 
    *, 
    cnp.ndarray[cnp.intp_t, ndim = 1] spike_positions, 
    cnp.ndarray[cnp.intp_t, ndim = 1] channels
):
    """
    Subtract the average waveform of detected spikes from the extracellular recording.

    This function:
    - Extracts a 4ms window around each detected spike across selected channels.
    - Accumulates and averages these waveforms across all detected spikes.
    - Subtracts the average waveform in-place from the recording.
    - Reduces overlap artifacts and improves detection of weak spikes.

    Parameters
    ----------
    data : ndarray of shape (num_samples, recording_channels, samples_per_ms), dtype=float32
        Extracellular recording array. Modified in-place.
    spike_positions : ndarray of shape (num_spikes,), dtype=intp
        Detected spike positions (in sample units).
    channels : ndarray of shape (num_channels,), dtype=intp
        Indices of channels to subtract from.

    Returns
    -------
    None
        The input data is modified in-place.
    """

    # Extract dimensions
    cdef cnp.intp_t num_spikes = spike_positions.shape[0]
    cdef cnp.intp_t num_channels = channels.shape[0]
    cdef cnp.intp_t recording_channels = data.shape[1]
    cdef cnp.intp_t samples_per_ms = data.shape[2]

    # Compute flattened memory block sizes
    cdef cnp.intp_t block_size = recording_channels * samples_per_ms
    cdef cnp.intp_t waveform_length = 4 * samples_per_ms

    # Pointer to data for direct memory access
    cdef cnp.float32_t *data_ptr = &data[0,0,0]

    # Allocate average waveform buffer
    cdef cnp.ndarray[cnp.float32_t, ndim = 2] avg_waveform = np.zeros((num_channels, waveform_length), dtype = 'float32')
    cdef cnp.float32_t *avg_waveform_ptr = &avg_waveform[0,0]

    # Declare loop variables
    cdef cnp.intp_t spike_idx          # Index over spikes
    cdef cnp.intp_t channel_offset     # Index over selected channels
    cdef cnp.intp_t channel_idx        # Absolute channel index
    cdef cnp.intp_t sample_idx         # Index over samples within 1 ms
    cdef cnp.intp_t ms_idx             # Whole millisecond index
    cdef cnp.intp_t offset             # Sub-ms sample index within millisecond
    cdef cnp.intp_t data_idx           # Linearized index into data_ptr
    cdef cnp.intp_t avg_idx = 0        # Flat write position in waveform array

    # Step 1: Accumulate waveforms
    for spike_idx in range(num_spikes):
        # Compute ms block and sample offset within that block
        ms_idx = spike_positions[spike_idx] // samples_per_ms
        offset = spike_positions[spike_idx] % samples_per_ms
        avg_idx = 0

        # Loop through selected channels (e.g., spatial neighbors)
        for channel_offset in range(num_channels):
            channel_idx = channels[channel_offset]

            # Compute base offset to current (ms, channel)
            data_idx = samples_per_ms * (
                ms_idx * recording_channels + channel_idx
            )

            # 1. Pre-spike: trailing samples of -2ms
            for sample_idx in range(offset, samples_per_ms):
                avg_waveform_ptr[avg_idx] += data_ptr[data_idx + sample_idx - block_size * 2]
                avg_idx += 1

            # 2. Pre-spike: -1ms
            for sample_idx in range(samples_per_ms):
                avg_waveform_ptr[avg_idx] += data_ptr[data_idx + sample_idx - block_size]
                avg_idx += 1

            # 3. Spike-centered: 0ms
            for sample_idx in range(samples_per_ms):
                avg_waveform_ptr[avg_idx] += data_ptr[data_idx + sample_idx]
                avg_idx += 1

            # 4. Post-spike: +1ms
            for sample_idx in range(samples_per_ms):
                avg_waveform_ptr[avg_idx] += data_ptr[data_idx + sample_idx + block_size]
                avg_idx += 1

            # 5. Post-spike: leading samples of +2ms
            for sample_idx in range(offset):
                avg_waveform_ptr[avg_idx] += data_ptr[data_idx + sample_idx + block_size * 2]
                avg_idx += 1

    # Step 2: Normalize to get average waveform
    for avg_idx in range(num_channels * waveform_length):
        avg_waveform_ptr[avg_idx] /= num_spikes

    # Step 3: Subtract average waveform in-place
    for spike_idx in range(num_spikes):
        # Compute ms block and sample offset within that block
        ms_idx = spike_positions[spike_idx] // samples_per_ms
        offset = spike_positions[spike_idx] % samples_per_ms
        avg_idx = 0

        # Loop through selected channels (e.g., spatial neighbors)
        for channel_offset in range(num_channels):
            channel_idx = channels[channel_offset]

            # Compute base offset to current (ms, channel)
            data_idx = samples_per_ms * (
                ms_idx * recording_channels + channel_idx
            )

            # 1. Pre-spike: trailing samples of -2ms
            for sample_idx in range(offset, samples_per_ms):
                data_ptr[data_idx + sample_idx - block_size * 2] -= avg_waveform_ptr[avg_idx]
                avg_idx += 1

            # 2. Pre-spike: -1ms
            for sample_idx in range(samples_per_ms):
                data_ptr[data_idx + sample_idx - block_size] -= avg_waveform_ptr[avg_idx]
                avg_idx += 1

            # 3. Spike-centered: 0ms
            for sample_idx in range(samples_per_ms):
                data_ptr[data_idx + sample_idx] -= avg_waveform_ptr[avg_idx]
                avg_idx += 1

            # 4. Post-spike: +1ms
            for sample_idx in range(samples_per_ms):
                data_ptr[data_idx + sample_idx + block_size] -= avg_waveform_ptr[avg_idx]
                avg_idx += 1

            # 5. Post-spike: leading samples of +2ms
            for sample_idx in range(offset):
                data_ptr[data_idx + sample_idx + block_size * 2] -= avg_waveform_ptr[avg_idx]
                avg_idx += 1

@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative indexing for performance
def update_spike_statistics(
    cnp.ndarray[cnp.float32_t, ndim = 3] data, 
    *, 
    cnp.ndarray[cnp.intp_t, ndim = 2] min_pos, 
    cnp.ndarray[cnp.float32_t, ndim = 2] min_values, 
    cnp.intp_t spike_position, 
    cnp.ndarray[cnp.intp_t, ndim = 1] channels, 
    cnp.ndarray[cnp.float32_t, ndim = 1] amplitude_scores, 
    cnp.ndarray[cnp.float32_t, ndim = 1] mad_thresholds
):
    """
    Update detection statistics after subtracting a spike waveform.

    This function:
    - Recomputes the minimum value and its index for each affected (channel, ms) entry.
    - Adjusts the amplitude scores by removing old and adding new contributions.
    - Operates within a 5ms window around the spike.

    Parameters
    ----------
    data : ndarray of shape (num_samples, recording_channels, samples_per_ms), dtype=float32
        The extracellular voltage recording.
    min_pos : ndarray of shape (recording_channels, num_milliseconds), dtype=intp
        Index of the minimum value per millisecond per channel.
    min_values : ndarray of shape (recording_channels, num_milliseconds), dtype=float32
        Value of the per-millisecond minimum per channel.
    spike_position : int
        Spike occurrence time in milliseconds.
    channels : ndarray of shape (num_channels,), dtype=intp
        Affected channel indices (usually spatially local).
    amplitude_scores : ndarray of shape (recording_channels,), dtype=float32
        Cumulative deviation below MAD for each channel.
    mad_thresholds : ndarray of shape (recording_channels,), dtype=float32
        Noise thresholds used for spike detection (per channel MAD).

    Returns
    -------
    None
        The function updates min_pos, min_values, and amplitude_scores in-place.
    """

    # Get recording dimensions
    cdef cnp.intp_t num_channels = channels.shape[0]
    cdef cnp.intp_t num_milliseconds = data.shape[0]
    cdef cnp.intp_t recording_channels = data.shape[1]
    cdef cnp.intp_t samples_per_ms = data.shape[2]

    # Pointer to raw data for direct memory access
    cdef cnp.float32_t *data_ptr = &data[0,0,0]

    # Declare loop variables
    cdef cnp.intp_t ms_idx           # millisecond index
    cdef cnp.intp_t channel_offset   # Index over selected channels
    cdef cnp.intp_t channel_idx      # channel index
    cdef cnp.intp_t sample_idx       # sample index within millisecond
    cdef cnp.intp_t data_idx         # index in flattened input array
    cdef cnp.intp_t min_position     # index of minimum value within current ms block
    cdef cnp.float32_t min_value     # value of the minimum

    # Update each affected (channel, ms) pair
    for channel_offset in range(num_channels):
        channel_idx = channels[channel_offset]
        for ms_idx in range(spike_position-2, spike_position+3):
            # Step 1: Remove previous min contribution from amplitude_scores
            min_value = min_values[channel_idx, ms_idx]
            amplitude_scores[channel_idx] -= min(0.0,
                min_value - mad_thresholds[channel_idx]
            )

            # Step 2: Recompute the new minimum in the affected ms/channel
            data_idx = samples_per_ms * (
                ms_idx * recording_channels + channel_idx
            )
            # ensure first value always gets checked
            min_value = data_ptr[data_idx] + 1
            min_position = 0

            for sample_idx in range(samples_per_ms):
                # Track minimum value and its index
                if data_ptr[data_idx] < min_value:
                    min_value = data_ptr[data_idx]
                    min_position = sample_idx
                data_idx += 1

            # Step 3: Store new min position and value
            min_pos[channel_idx, ms_idx] = min_position
            min_values[channel_idx, ms_idx] = min_value

            # Step 4: Add updated min contribution to amplitude_scores
            amplitude_scores[channel_idx] += min(0.0,
                min_value - mad_thresholds[channel_idx]
            )
