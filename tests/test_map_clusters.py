import unittest
import numpy as np
import tempfile
import os
from spikesift import Recording, perform_spike_sorting, merge_recordings, map_clusters

class TestMapClusters(unittest.TestCase):

    def _create_temp_recording(self, *, num_channels, num_samples, dtype="int16"):
        signal = (np.random.randn(num_samples, num_channels) * 5).astype(dtype)
        binary_data = signal.ravel().tobytes()
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(binary_data)
        f.close()

        probe_geometry = np.stack([
            np.zeros(num_channels, dtype=np.float32),
            np.arange(num_channels, dtype=np.float32) * 25
        ], axis=1)

        recording = Recording(
            binary_file=f.name,
            data_type=dtype,
            probe_geometry=probe_geometry,
            sampling_frequency=20000,
        )
        return recording, f.name

    def test_map_clusters_valid_mapping(self):
        num_channels = 8
        num_samples = 8000

        rec1, path1 = self._create_temp_recording(num_channels=num_channels, num_samples=num_samples)
        rec2, path2 = self._create_temp_recording(num_channels=num_channels, num_samples=num_samples)

        try:
            rec2.recording_offset = rec1.num_samples

            sorted1 = perform_spike_sorting(
                rec1, min_segment_length=0.2, detection_sensitivity=6, verbose=False
            )
            sorted2 = perform_spike_sorting(
                rec2, min_segment_length=0.2, detection_sensitivity=6, verbose=False
            )

            # Merge to ensure alignment is meaningful
            merged = merge_recordings([sorted1, sorted2])
            split = merged.split_into_segments()

            self.assertEqual(len(split), 2)

            cluster_map = map_clusters(split[0], split[1])
            self.assertIsInstance(cluster_map, dict)
            for src_id, tgt_id in cluster_map.items():
                self.assertIsInstance(src_id, int)
                self.assertIsInstance(tgt_id, int)
                self.assertTrue(split[0].valid_cluster_id(src_id))
                self.assertTrue(split[1].valid_cluster_id(tgt_id))

        finally:
            os.remove(path1)
            os.remove(path2)

    def test_map_clusters_mismatched_geometry_raises(self):
        rec1, path1 = self._create_temp_recording(num_channels=8, num_samples=8000)
        rec2, path2 = self._create_temp_recording(num_channels=8, num_samples=8000)

        try:
            rec2.recording_offset = rec1.num_samples

            sorted1 = perform_spike_sorting(rec1, min_segment_length=0.2, detection_sensitivity=6, verbose=False)
            sorted2 = perform_spike_sorting(rec2, min_segment_length=0.2, detection_sensitivity=6, verbose=False)

            # Tamper with geometry
            sorted2.probe_geometry[0, 1] += 1.0

            with self.assertRaises(ValueError) as context:
                map_clusters(sorted1, sorted2)
            self.assertIn("must match exactly", str(context.exception))

        finally:
            os.remove(path1)
            os.remove(path2)

    def test_map_clusters_invalid_inputs(self):
        rec, path = self._create_temp_recording(num_channels=8, num_samples=8000)
        try:
            sorted_rec = perform_spike_sorting(rec, min_segment_length=0.2, detection_sensitivity=6, verbose=False)

            with self.assertRaises(ValueError):
                map_clusters("not_a_sorted_recording", sorted_rec)

            with self.assertRaises(ValueError):
                map_clusters(sorted_rec, "still_not_valid")

        finally:
            os.remove(path)

if __name__ == "__main__":
    unittest.main()