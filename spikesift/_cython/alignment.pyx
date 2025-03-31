import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def simulate_drift(
    cnp.ndarray[cnp.float32_t, ndim = 2] amplitude_vectors, 
    *, 
    cnp.ndarray[cnp.float32_t, ndim = 2] probe_geometry, 
    cnp.float32_t step
):
    """
    Simulate vertical drift by interpolating amplitude vectors along the probe axis.

    This function:
    - Offsets each recording site's y-coordinate by a given vertical step (in micrometers).
    - For each shifted site, finds the nearest two vertically aligned electrodes (within 2 um lateral distance).
    - Computes linear interpolation weights based on vertical distances.
    - Uses fallback to nearest neighbor if interpolation is not possible.

    Parameters
    ----------
    amplitude_vectors : ndarray of shape (num_units, recording_channels), dtype=float32
        Amplitude profiles for each detected unit (rows).
    probe_geometry : ndarray of shape (recording_channels, 2), dtype=float32
        2D (x, y) coordinates of each electrode site.
    step : float
        Vertical drift offset in micrometers to apply to each channel's y-position.

    Returns
    -------
    shifted_vectors : ndarray of shape (num_units, recording_channels), dtype=float32
        Amplitude vectors interpolated to simulate vertical drift.
    """

    # Number of recording channels
    cdef cnp.intp_t recording_channels = probe_geometry.shape[0]

    # Sort channels by vertical position (y-axis)
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] sorted_indices = np.argsort(probe_geometry[:, 1])
    cdef cnp.ndarray[cnp.float32_t, ndim = 2] sorted_probe = probe_geometry[sorted_indices]
    cdef cnp.float32_t *sorted_ptr = &sorted_probe[0, 0]

    # Output arrays for interpolation indices and weights
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] ch1 = np.empty(recording_channels, dtype = 'intp')
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] ch2 = np.empty(recording_channels, dtype = 'intp')
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] w1 = np.empty(recording_channels, dtype = 'float32')
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] w2 = np.empty(recording_channels, dtype = 'float32')

    # Declare loop variables
    cdef cnp.intp_t channel_idx      # Index into sorted geometry
    cdef cnp.intp_t sorted_idx = 0   # Vertical scan index
    cdef cnp.intp_t left_idx         # Lower y neighbor in sorted geometry
    cdef cnp.intp_t right_idx        # Upper y neighbor in sorted geometry
    cdef cnp.intp_t orig_idx         # Original channel index
    cdef cnp.float32_t x             # X position of current channel
    cdef cnp.float32_t y_shifted     # Y position after applying vertical drift
    cdef cnp.float32_t dy1, dy2      # Distances to interpolation neighbors

    # Loop over channels to simulate drift
    for channel_idx in range(recording_channels):
        # Extract x, y and apply drift
        x = sorted_ptr[channel_idx * 2]
        y_shifted = sorted_ptr[channel_idx * 2 + 1] + step

        # Find the vertical insertion index in sorted geometry
        while sorted_idx < recording_channels and (
            sorted_ptr[2 * sorted_idx + 1] < y_shifted
        ):
            sorted_idx += 1

        # Search for valid left neighbor (below y_shifted)
        left_idx = sorted_idx - 1
        while left_idx > 0 and (
            abs(sorted_ptr[2 * left_idx] - x) > 2
        ):
            left_idx -= 1

        # Search for valid right neighbor (above y_shifted)
        right_idx = sorted_idx
        while right_idx < recording_channels and (
            abs(sorted_ptr[2 * right_idx] - x) > 2
        ):
            right_idx += 1

        # Handle edge cases: fallback to nearest neighbor
        if left_idx == -1:
            left_idx, dy1, dy2 = right_idx, 1.0, 1.0
        elif right_idx == recording_channels:
            right_idx, dy1, dy2 = left_idx, 1.0, 1.0
        else:
            dy1 = y_shifted - sorted_ptr[2 * left_idx + 1]
            dy2 = sorted_ptr[2 * right_idx + 1] - y_shifted

        # Map back to original channel index
        orig_idx = sorted_indices[channel_idx]
        ch1[orig_idx] = sorted_indices[left_idx]
        ch2[orig_idx] = sorted_indices[right_idx]
        w1[orig_idx] = dy2 / (dy1 + dy2)
        w2[orig_idx] = dy1 / (dy1 + dy2)

    # Apply linear interpolation of amplitudes using computed weights
    return amplitude_vectors[:, ch1] * w1 + amplitude_vectors[:, ch2] * w2