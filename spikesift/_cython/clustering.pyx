import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative indexing for performance
def compute_optimal_threshold(
    cnp.ndarray[cnp.float32_t, ndim = 1] values
):
    """
    Compute a separation threshold using a nearest-neighbor variance merging approach.

    This function:
    - Treats the sorted 1D input values as candidate spike features.
    - Iteratively merges adjacent values with minimal squared distance,
      weighted by cluster size, to reduce within-cluster variance.
    - Continues merging until two clusters remain.
    - Returns the boundary value separating the two final clusters.

    This is a fast 1D clustering heuristic inspired by Ward's method.

    Parameters
    ----------
    values : ndarray of shape (num_samples,), dtype=float32
        Sorted array of scalar feature values (e.g., voltage, PCA score).

    Returns
    -------
    threshold : float
        Estimated boundary between the two most distinct clusters.
    """

    # Get the number of input values
    cdef cnp.intp_t num_values = values.shape[0]

    # Create a working copy of input values for in-place modification
    cdef cnp.float32_t[:] working_values = values.copy()

    # Cluster size for each cluster (all start as 1)
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] cluster_size = np.ones(num_values, dtype = 'intp')
    cdef cnp.intp_t *cluster_size_ptr = &cluster_size[0]

    # Linked list to track next neighbors for merging
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] next_neighbor = np.arange(num_values, dtype = 'intp') + 1
    cdef cnp.intp_t *next_neighbor_ptr = &next_neighbor[0]

    # Linked list to track previous neighbors
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] prev_neighbor = np.arange(num_values, dtype = 'intp') - 1
    cdef cnp.intp_t *prev_neighbor_ptr = &prev_neighbor[0]

    # Initialize index pointer for merging
    cdef cnp.intp_t idx = 1

    # Perform num_values - 2 merge operations (stop when 2 clusters remain)
    for _ in range(num_values - 2):
        # Find two mutual nearest neighbors
        while (
            next_neighbor[idx] < num_values and
            (working_values[idx] - working_values[prev_neighbor[idx]]) ** 2 * min(cluster_size[idx], cluster_size[prev_neighbor[idx]]) >
            (working_values[next_neighbor[idx]] - working_values[idx]) ** 2 * min(cluster_size[idx], cluster_size[next_neighbor[idx]])
        ):
            idx = next_neighbor[idx]  # Move to next neighbor

        # Update linked list pointers after merge
        if next_neighbor[idx] < num_values:
            prev_neighbor[next_neighbor[idx]] = prev_neighbor[idx]
        next_neighbor[prev_neighbor[idx]] = next_neighbor[idx]

        # Update merged cluster size
        cluster_size[prev_neighbor[idx]] += cluster_size[idx]

        # Weighted mean update of cluster center
        working_values[prev_neighbor[idx]] += (
            (working_values[idx] - working_values[prev_neighbor[idx]]) * 
            cluster_size[idx] / cluster_size[prev_neighbor[idx]]
        )

        # Move idx back to previous neighbor (skipping dead entries)
        idx = prev_neighbor[prev_neighbor[idx]]
        if idx <= 0: idx = next_neighbor[0]

    # Return the boundary between the two final clusters
    return values[next_neighbor[0]]

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_principal_projection(
    cnp.ndarray[cnp.float32_t, ndim = 2] data
):
    """
    Project the dataset onto its principal direction using power iteration.

    This function:
    - Computes the top eigenvector of the data covariance matrix using 10-step power iteration.
    - Handles both tall (samples >= features) and wide (features > samples) matrices.
    - Normalizes every two steps to avoid numerical blow-up or vanishing.
    - Returns a 1D projection vector capturing maximum variance across samples.

    Parameters
    ----------
    data : ndarray of shape (num_samples, num_features), dtype=float32
        Input matrix where each row is a sample and each column is a feature.

    Returns
    -------
    projected_data : ndarray of shape (num_samples,), dtype=float32
        1D projection of all samples onto the principal direction.
    """

    # Extract input dimensions
    cdef int num_samples = data.shape[0]
    cdef int num_features = data.shape[1]

    # Compute L2 norm per sample and select the largest as initial vector
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] sample_norms = np.linalg.norm(data, axis = 1)
    cdef cnp.intp_t max_norm_idx = np.argmax(sample_norms)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] principal_vector = data[max_norm_idx].copy()

    # Declare variables used in both cases
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] projected_data
    cdef cnp.ndarray[cnp.float32_t, ndim = 2] covariance
    cdef cnp.float32_t norm

    if num_samples < num_features:
        # Case 1: Wide matrix (samples < features)
        # Compute covariance in sample space and iterate on projections
        projected_data = data @ principal_vector
        covariance = data @ data.T

        for _ in range(5):
            # Two-step power iteration with normalization for stability
            projected_data = covariance @ projected_data
            norm = np.linalg.norm(projected_data)
            if norm > 1e-8: 
                projected_data *= 1 / norm
            projected_data = covariance @ projected_data

        return projected_data

    else:
        # Case 2: Tall matrix (samples >= features)
        # Iterate on the principal direction in feature space
        covariance = data.T @ data

        for _ in range(5):
            # Two-step power iteration with normalization for stability
            principal_vector = covariance @ principal_vector
            norm = np.linalg.norm(principal_vector)
            if norm > 1e-8: 
                principal_vector *= 1 / norm
            principal_vector = covariance @ principal_vector

        return data @ principal_vector

@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative indexing for performance
def compute_inter_channel_differences(
    cnp.ndarray[cnp.float32_t, ndim = 2] waveform
):
    """
    Compute inter-channel voltage differences for robust clustering.

    This function:
    - Computes pairwise directional voltage differences between channels.
    - Designed to detect spatial separation of spikes across electrodes.
    - Produces a 20-element feature vector for either 5-channel or 4-channel input.
    - In 5-channel mode, returns 20 pairwise differences (all ordered pairs).
    - In 4-channel mode, fills the last 8 values with per-channel max abs values.

    Parameters
    ----------
    waveform : ndarray of shape (num_channels, num_samples), dtype=float32
        Extracted waveform from a single spike. Channels are assumed stacked vertically.

    Returns
    -------
    inter_channel_diff : ndarray of shape (20,), dtype=float32
        Per-channel-pair voltage differences used for clustering or alignment.
    """

    # Extract waveform dimensions
    cdef cnp.intp_t num_channels = waveform.shape[0]
    cdef cnp.intp_t num_samples = waveform.shape[1] 

    # Flatten waveform memory layout for pointer-based access
    cdef cnp.float32_t *waveform_ptr = &waveform[0, 0]  

    # Allocate output vector for inter-channel differences
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] diff_vector = np.zeros(20, dtype = 'float32')
    cdef cnp.float32_t *diff_vector_ptr = &diff_vector[0]

    # Declare loop variable
    cdef cnp.intp_t sample_idx

    # Channel 0 vs Channel 1 
    for sample_idx in range(num_samples*1, num_samples*2):
        diff_vector_ptr[0] = max(diff_vector_ptr[0],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples])
        diff_vector_ptr[1] = max(diff_vector_ptr[1],waveform_ptr[sample_idx - num_samples] - waveform_ptr[sample_idx])

    # Channel 0/1 vs Channel 2 
    for sample_idx in range(num_samples*2, num_samples*3):
        diff_vector_ptr[2] = max(diff_vector_ptr[2],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples * 2])
        diff_vector_ptr[3] = max(diff_vector_ptr[3],waveform_ptr[sample_idx - num_samples*2] - waveform_ptr[sample_idx])
        diff_vector_ptr[4] = max(diff_vector_ptr[4],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples])
        diff_vector_ptr[5] = max(diff_vector_ptr[5],waveform_ptr[sample_idx - num_samples] - waveform_ptr[sample_idx])

    # Channel 0/1/2 vs Channel 3 
    for sample_idx in range(num_samples*3, num_samples*4):
        diff_vector_ptr[6] = max(diff_vector_ptr[6],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples * 3])
        diff_vector_ptr[7] = max(diff_vector_ptr[7],waveform_ptr[sample_idx - num_samples * 3] - waveform_ptr[sample_idx])
        diff_vector_ptr[8] = max(diff_vector_ptr[8],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples * 2])
        diff_vector_ptr[9] = max(diff_vector_ptr[9],waveform_ptr[sample_idx - num_samples * 2] - waveform_ptr[sample_idx])
        diff_vector_ptr[10] = max(diff_vector_ptr[10],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples])
        diff_vector_ptr[11] = max(diff_vector_ptr[11],waveform_ptr[sample_idx - num_samples] - waveform_ptr[sample_idx])

    if num_channels == 5:
        # If 5 channels: Channel 0/1/2/3 vs Channel 4 
        for sample_idx in range(num_samples*4, num_samples*5):
            diff_vector_ptr[12] = max(diff_vector_ptr[12],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples * 4])
            diff_vector_ptr[13] = max(diff_vector_ptr[13],waveform_ptr[sample_idx - num_samples * 4] - waveform_ptr[sample_idx])
            diff_vector_ptr[14] = max(diff_vector_ptr[14],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples * 3])
            diff_vector_ptr[15] = max(diff_vector_ptr[15],waveform_ptr[sample_idx - num_samples * 3] - waveform_ptr[sample_idx])
            diff_vector_ptr[16] = max(diff_vector_ptr[16],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples * 2])
            diff_vector_ptr[17] = max(diff_vector_ptr[17],waveform_ptr[sample_idx - num_samples * 2] - waveform_ptr[sample_idx])
            diff_vector_ptr[18] = max(diff_vector_ptr[18],waveform_ptr[sample_idx] - waveform_ptr[sample_idx - num_samples])
            diff_vector_ptr[19] = max(diff_vector_ptr[19],waveform_ptr[sample_idx - num_samples] - waveform_ptr[sample_idx])

    else: 
        # Fallback for 4-channel input: fill final 8 values with max absolute amplitude per channel 
        for sample_idx in range(num_samples):
            diff_vector_ptr[12] = max(diff_vector_ptr[12], -waveform_ptr[sample_idx])
            diff_vector_ptr[13] = max(diff_vector_ptr[13], waveform_ptr[sample_idx])
            diff_vector_ptr[14] = max(diff_vector_ptr[14], -waveform_ptr[sample_idx + num_samples])
            diff_vector_ptr[15] = max(diff_vector_ptr[15], waveform_ptr[sample_idx + num_samples])
            diff_vector_ptr[16] = max(diff_vector_ptr[16], -waveform_ptr[sample_idx + num_samples * 2])
            diff_vector_ptr[17] = max(diff_vector_ptr[17], waveform_ptr[sample_idx + num_samples * 2])
            diff_vector_ptr[18] = max(diff_vector_ptr[18], -waveform_ptr[sample_idx + num_samples * 3])
            diff_vector_ptr[19] = max(diff_vector_ptr[19], waveform_ptr[sample_idx + num_samples * 3])
        
    return diff_vector

@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative indexing for performance
def perform_hierarchical_clustering(
    cnp.ndarray[cnp.float32_t, ndim = 2] data,
    *,
    cnp.ndarray[cnp.intp_t, ndim = 1] spike_positions,
    cnp.ndarray[cnp.float32_t, ndim = 1] mean_waveform,
    cnp.intp_t samples_per_ms,
    cnp.float32_t min_spikes_per_cluster,
    cnp.float32_t merging_threshold,
    cnp.intp_t num_channels
):
    """
    Recursively refine spike clusters using binary splits and inter-channel separation.

    This function:
    - Projects the data along its principal direction using power iteration.
    - Splits it into two binary clusters using a 1D threshold heuristic.
    - Computes extrapolated means and inter-channel differences for each cluster.
    - If the clusters are sufficiently distinct, recursively refines the most negative one.
    - If the split fails, returns the input spike group as-is.

    Used to isolate single-unit spike clusters from noisy multi-unit activity.

    Parameters
    ----------
    data : ndarray of shape (num_spikes, num_features), dtype=float32
        Input spike feature matrix.
    spike_positions : ndarray of shape (num_spikes,), dtype=intp
        Detected spike positions (in sample units).
    mean_waveform : ndarray of shape (num_features,), dtype=float32
        Mean waveform of the parent cluster.
    samples_per_ms : int
        Number of samples per millisecond (used to index peak offset).
    min_spikes_per_cluster : float
        Minimum number of spikes required to keep a refined cluster.
    merging_threshold : float
        Threshold controlling how distinct the refined clusters must be.
    num_channels : int
        Number of channels used in waveform extraction (required for reshaping).

    Returns
    -------
    refined_positions : ndarray of shape (num_selected_spikes,), dtype=intp
        Spike times of the most prominent subcluster after refinement.
    refined_mean_waveform : ndarray of shape (num_features,), dtype=float32
        Average waveform of the selected subcluster.
    """

    # Base case: return if not enough spikes
    cdef cnp.intp_t num_spikes = data.shape[0]
    if num_spikes < min_spikes_per_cluster:
        return spike_positions, mean_waveform

    # Project along principal direction (1D PCA)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] principal_projection = compute_principal_projection(data - mean_waveform)

    # Compute optimal binary threshold in projected space
    cdef cnp.float32_t optimal_threshold = compute_optimal_threshold(np.sort(principal_projection))
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] cluster_labels = (principal_projection < optimal_threshold).astype('intp')

    # Count spikes in each subcluster
    cdef cnp.intp_t num_spikes_1 = cluster_labels.sum()
    cdef cnp.intp_t num_spikes_0 = num_spikes - num_spikes_1

    # Split failed: all points on one side
    if 0 in (num_spikes_0, num_spikes_1):
        return spike_positions, mean_waveform

    # Estimate extrapolated cluster means
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] mean0
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] mean1
    cdef cnp.float32_t factor

    if num_spikes_1 < num_spikes_0:
        mean1 = data[cluster_labels == 1].mean(axis = 0)
        factor = num_spikes_1 / num_spikes_0
        mean0 = mean_waveform + (mean_waveform - mean1) * factor
    else:
        mean0 = data[cluster_labels == 0].mean(axis = 0)
        factor = num_spikes_0 / num_spikes_1
        mean1 = mean_waveform + (mean_waveform - mean0) * factor

    # Compute inter-channel differences for both candidates
    cdef cnp.ndarray[cnp.float32_t, ndim=1] inter_channel_diff_0 = compute_inter_channel_differences(mean0.reshape(num_channels, -1))
    cdef cnp.ndarray[cnp.float32_t, ndim=1] inter_channel_diff_1 = compute_inter_channel_differences(mean1.reshape(num_channels, -1))

    # Compute separation score
    cdef cnp.float32_t xx = inter_channel_diff_0 @ inter_channel_diff_0
    cdef cnp.float32_t xy = inter_channel_diff_0 @ inter_channel_diff_1
    cdef cnp.float32_t yy = inter_channel_diff_1 @ inter_channel_diff_1

    # Reject split if too similar
    if xx - 2 * xy + yy < merging_threshold ** 2 * max(xx, yy):
        return spike_positions, mean_waveform

    # Recursive refinement
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] refined_positions
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] refined_mean_waveform

    if mean0[samples_per_ms] < mean1[samples_per_ms]:
        refined_positions, refined_mean_waveform = perform_hierarchical_clustering(
            data[cluster_labels == 0],
            spike_positions = spike_positions[cluster_labels == 0],
            mean_waveform = mean0,
            samples_per_ms = samples_per_ms,
            min_spikes_per_cluster = min_spikes_per_cluster,
            merging_threshold = merging_threshold,
            num_channels = num_channels
        )
        # If refinement failed, try the other cluster
        if refined_positions.shape[0] < min_spikes_per_cluster:
            return perform_hierarchical_clustering(
                data[cluster_labels == 1],
                spike_positions = spike_positions[cluster_labels == 1],
                mean_waveform = mean1,
                samples_per_ms = samples_per_ms,
                min_spikes_per_cluster = min_spikes_per_cluster,
                merging_threshold = merging_threshold,
                num_channels = num_channels
            )
    else:
        refined_positions, refined_mean_waveform = perform_hierarchical_clustering(
            data[cluster_labels == 1],
            spike_positions = spike_positions[cluster_labels == 1],
            mean_waveform = mean1,
            samples_per_ms = samples_per_ms,
            min_spikes_per_cluster = min_spikes_per_cluster,
            merging_threshold = merging_threshold,
            num_channels = num_channels
        )
        # If refinement failed, try the other cluster
        if refined_positions.shape[0] < min_spikes_per_cluster:
            return perform_hierarchical_clustering(
                data[cluster_labels == 0],
                spike_positions = spike_positions[cluster_labels == 0],
                mean_waveform = mean0,
                samples_per_ms = samples_per_ms,
                min_spikes_per_cluster = min_spikes_per_cluster,
                merging_threshold = merging_threshold,
                num_channels = num_channels
            )
    
    return refined_positions, refined_mean_waveform

@cython.boundscheck(False)  # Turn off bounds-checking for performance
@cython.wraparound(False)   # Turn off negative indexing for performance
def perform_final_clustering(
    cnp.ndarray[cnp.float32_t, ndim = 2] data,
    *,
    cnp.ndarray[cnp.intp_t, ndim = 1] spike_positions,
    cnp.ndarray[cnp.float32_t, ndim = 1] mean_waveform,
    cnp.ndarray[cnp.float32_t, ndim = 1] reference_vector,
    cnp.float32_t min_spikes_per_cluster,
    cnp.float32_t merging_threshold,
    cnp.intp_t num_channels
):
    """
    Final clustering stage using recursive binary splits and reference alignment.

    This function:
    - Projects spike features along the principal direction of variance.
    - Performs a binary split using an optimal 1D threshold.
    - Computes inter-channel differences for both clusters.
    - Chooses the cluster most similar to a given reference waveform.
    - Recurses until no further valid splits are found.

    Used to isolate a single, stable unit from a multi-unit spike cluster.

    Parameters
    ----------
    data : ndarray of shape (num_spikes, num_features), dtype=float32
        Spike feature matrix.
    spike_positions : ndarray of shape (num_spikes,), dtype=intp
        Timestamps of detected spikes (sample indices).
    mean_waveform : ndarray of shape (num_features,), dtype=float32
        Mean waveform of the current cluster.
    reference_vector : ndarray of shape (num_features,), dtype=float32
        Fixed target waveform (e.g., from initial clustering) for alignment.
    min_spikes_per_cluster : float
        Minimum number of spikes required to keep a cluster.
    merging_threshold : float
        Squared separation threshold for stopping the split.
    num_channels : int
        Number of waveform channels (used to reshape waveforms).

    Returns
    -------
    refined_positions : ndarray of shape (num_selected_spikes,), dtype=intp
        Selected spike times for the refined cluster.
    refined_mean_waveform : ndarray of shape (num_features,), dtype=float32
        Mean waveform of the final selected cluster.
    """

    # Base case: return if not enough spikes
    cdef cnp.intp_t num_spikes = data.shape[0]
    if num_spikes < min_spikes_per_cluster:
        return spike_positions, mean_waveform

    # Project along principal direction (1D PCA)
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] principal_projection = compute_principal_projection(data - mean_waveform)

    # Compute optimal binary threshold in projected space
    cdef cnp.float32_t optimal_threshold = compute_optimal_threshold(np.sort(principal_projection))
    cdef cnp.ndarray[cnp.intp_t, ndim = 1] cluster_labels = (principal_projection < optimal_threshold).astype('intp')

    # Count spikes in each subcluster
    cdef cnp.intp_t num_spikes_1 = cluster_labels.sum()
    cdef cnp.intp_t num_spikes_0 = num_spikes - num_spikes_1

    # Split failed: all points on one side
    if 0 in (num_spikes_0, num_spikes_1):
        return spike_positions, mean_waveform

    # Estimate extrapolated cluster means
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] mean0
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] mean1
    cdef cnp.float32_t factor

    if num_spikes_1 < num_spikes_0:
        mean1 = data[cluster_labels == 1].mean(axis = 0)
        factor = num_spikes_1 / num_spikes_0
        mean0 = mean_waveform + (mean_waveform - mean1) * factor
    else:
        mean0 = data[cluster_labels == 0].mean(axis = 0)
        factor = num_spikes_0 / num_spikes_1
        mean1 = mean_waveform + (mean_waveform - mean0) * factor

    # Compute inter-channel differences for both candidates
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] inter_channel_diff_0 = compute_inter_channel_differences(mean0.reshape(num_channels, -1))
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] inter_channel_diff_1 = compute_inter_channel_differences(mean1.reshape(num_channels, -1))

    # Compute separation score
    cdef cnp.float32_t xx = inter_channel_diff_0 @ inter_channel_diff_0
    cdef cnp.float32_t xy = inter_channel_diff_0 @ inter_channel_diff_1
    cdef cnp.float32_t yy = inter_channel_diff_1 @ inter_channel_diff_1

    # Reject split if too similar
    if xx - 2 * xy + yy < merging_threshold ** 2 * max(xx, yy):
        return spike_positions, mean_waveform

    # Choose cluster more aligned with reference vector
    if xx - 2 * reference_vector @ inter_channel_diff_0 < yy - 2 * reference_vector @ inter_channel_diff_1:
        return perform_final_clustering(
            data[cluster_labels == 0],
            spike_positions = spike_positions[cluster_labels == 0],
            mean_waveform = mean0,
            reference_vector = reference_vector,
            min_spikes_per_cluster = min_spikes_per_cluster,
            merging_threshold = merging_threshold,
            num_channels = num_channels
        )
    else:
        return perform_final_clustering(
            data[cluster_labels == 1],
            spike_positions = spike_positions[cluster_labels == 1],
            mean_waveform = mean1,
            reference_vector = reference_vector,
            min_spikes_per_cluster = min_spikes_per_cluster,
            merging_threshold = merging_threshold,
            num_channels = num_channels
        )
