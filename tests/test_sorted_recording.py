import unittest
import numpy as np
import tempfile
import os
from spikesift import Recording, perform_spike_sorting

class TestSortedRecording(unittest.TestCase):

    def _generate_synthetic_recording(self, *, num_channels, num_samples, dtype="int16"):
        # Create fake raw voltage signal with small noise
        signal = (np.random.randn(num_samples, num_channels) * 10).astype(dtype)
        binary_data = signal.ravel().tobytes()

        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(binary_data)
        f.close()

        probe_geometry = np.zeros((num_channels, 2), dtype=np.float32)
        for i in range(num_channels):
            probe_geometry[i] = [0, i * 20]  # 20 µm spacing vertically

        recording = Recording(
            binary_file=f.name,
            data_type=dtype,
            probe_geometry=probe_geometry,
            sampling_frequency=20000,
        )
        return recording, f.name

    def test_sorted_recording_end_to_end(self):
        num_channels = 8
        num_samples = 10000  # 0.5s at 20kHz

        recording, path = self._generate_synthetic_recording(
            num_channels=num_channels,
            num_samples=num_samples,
        )

        try:
            sorted_rec = perform_spike_sorting(
                recording,
                min_segment_length=0.2,  # force multiple segments
                detection_sensitivity=5,
                min_spikes_per_cluster=3,
                merging_threshold=0.3,
                max_drift=40,
                detection_polarity=-1,
                verbose=False,
            )

            # Should not raise errors
            segment_bounds = sorted_rec.segment_boundaries()
            self.assertGreaterEqual(len(segment_bounds), 1)

            cluster_ids = sorted_rec.cluster_ids()
            self.assertIsInstance(cluster_ids, set)

            for cid in cluster_ids:
                self.assertTrue(sorted_rec.valid_cluster_id(cid))
                spikes = sorted_rec.spikes(cid)
                amps = sorted_rec.amplitude_vectors(cid)
                self.assertIsInstance(spikes, np.ndarray)
                self.assertGreater(len(spikes), 0)
                self.assertEqual(amps.shape[1], num_channels)
                self.assertEqual(amps.shape[0], len(segment_bounds))

            all_spikes = sorted_rec.all_spikes()
            self.assertEqual(set(all_spikes.keys()), cluster_ids)

            split = sorted_rec.split_into_segments()
            self.assertEqual(len(split), len(segment_bounds))
            for sr in split:
                self.assertEqual(len(sr.sorted_segments), 1)
                self.assertEqual(sr.assignment_chain, [])
                self.assertEqual(sr.probe_geometry.shape[0], num_channels)

            self.assertEqual(len(sorted_rec), len(cluster_ids))
            self.assertGreaterEqual(sorted_rec.end_time(), sorted_rec.start_time())

        finally:
            os.remove(path)

    def test_sorted_recording_with_no_valid_clusters(self):
        num_channels = 8
        num_samples = 5000  # Short recording with minimal spikes

        recording, path = self._generate_synthetic_recording(
            num_channels=num_channels,
            num_samples=num_samples,
        )

        try:
            sorted_rec = perform_spike_sorting(
                recording,
                min_segment_length=0.3,
                detection_sensitivity=50,  # Very high threshold = likely no spikes
                min_spikes_per_cluster=10,
                verbose=False,
            )

            self.assertEqual(len(sorted_rec), 0)
            self.assertEqual(sorted_rec.cluster_ids(), set())
            self.assertEqual(sorted_rec.all_spikes(), {})

        finally:
            os.remove(path)

    def test_sorted_recording_too_short(self):
        num_channels = 8
        num_samples = 100  # 5 ms at 20kHz (too short)

        recording, path = self._generate_synthetic_recording(
            num_channels=num_channels,
            num_samples=num_samples,
        )

        try:
            with self.assertRaises(ValueError) as context:
                perform_spike_sorting(recording, verbose=False)

            self.assertIn("too short", str(context.exception))

        finally:
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
