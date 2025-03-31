import unittest
import numpy as np
import tempfile
import os
from spikesift import Recording, perform_spike_sorting, merge_recordings

class TestMergeRecordings(unittest.TestCase):

    def _create_temp_recording(self, *, num_channels, num_samples, dtype="int16"):
        # Generate random noise signal for spike sorting
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

    def test_merge_sorted_recordings(self):
        num_channels = 8
        samples_per_segment = 8000  # 0.4 sec per segment
        dtype = "int16"

        # Create two recordings with non-overlapping times
        rec1, path1 = self._create_temp_recording(num_channels=num_channels, num_samples=samples_per_segment, dtype=dtype)
        rec2, path2 = self._create_temp_recording(num_channels=num_channels, num_samples=samples_per_segment, dtype=dtype)

        try:
            # Apply spike sorting
            sorted1 = perform_spike_sorting(
                rec1,
                min_segment_length=0.2,
                detection_sensitivity=6,
                verbose=False
            )

            # Shift rec2's logical start time to avoid overlap
            rec2.recording_offset = rec1.num_samples
            sorted2 = perform_spike_sorting(
                rec2,
                min_segment_length=0.2,
                detection_sensitivity=6,
                verbose=False
            )

            # Merge both sorted recordings
            merged = merge_recordings([sorted1, sorted2], max_drift=35)

            # Verify merged recording spans both parts
            self.assertGreaterEqual(merged.end_time(), rec1.num_samples + rec2.num_samples)
            self.assertEqual(len(merged.sorted_segments), len(sorted1.sorted_segments) + len(sorted2.sorted_segments))

            # All cluster IDs in merged must be valid
            for cid in merged.cluster_ids():
                self.assertTrue(merged.valid_cluster_id(cid))
                spikes = merged.spikes(cid)
                amps = merged.amplitude_vectors(cid)
                self.assertGreater(len(spikes), 0)
                self.assertEqual(amps.shape[1], num_channels)

            # Confirm split segments match originals
            split = merged.split_into_segments()
            self.assertEqual(len(split), len(merged.sorted_segments))
            for sr in split:
                self.assertEqual(len(sr.sorted_segments), 1)
                self.assertEqual(sr.assignment_chain, [])

        finally:
            os.remove(path1)
            os.remove(path2)

    def test_merge_recordings_rejects_overlap(self):
        num_channels = 8
        samples_per_segment = 8000  # 0.4s
        dtype = "int16"

        rec1, path1 = self._create_temp_recording(num_channels=num_channels, num_samples=samples_per_segment, dtype=dtype)
        rec2, path2 = self._create_temp_recording(num_channels=num_channels, num_samples=samples_per_segment, dtype=dtype)

        try:
            sorted1 = perform_spike_sorting(
                rec1,
                min_segment_length=0.2,
                detection_sensitivity=6,
                verbose=False
            )
            # Do NOT shift rec2 — overlapping in time with rec1
            sorted2 = perform_spike_sorting(
                rec2,
                min_segment_length=0.2,
                detection_sensitivity=6,
                verbose=False
            )

            with self.assertRaises(ValueError) as context:
                merge_recordings([sorted1, sorted2])

            self.assertIn("overlaps with recording", str(context.exception))

        finally:
            os.remove(path1)
            os.remove(path2)

    def test_merge_recordings_rejects_mismatched_geometry(self):
        num_channels = 8
        samples_per_segment = 8000
        dtype = "int16"

        rec1, path1 = self._create_temp_recording(num_channels=num_channels, num_samples=samples_per_segment, dtype=dtype)
        rec2, path2 = self._create_temp_recording(num_channels=num_channels, num_samples=samples_per_segment, dtype=dtype)

        try:
            rec2.recording_offset = rec1.num_samples
            sorted1 = perform_spike_sorting(
                rec1,
                min_segment_length=0.2,
                detection_sensitivity=6,
                verbose=False
            )
            sorted2 = perform_spike_sorting(
                rec2,
                min_segment_length=0.2,
                detection_sensitivity=6,
                verbose=False
            )

            # Tamper with rec2 geometry
            sorted2.probe_geometry[0, 1] += 1.0

            with self.assertRaises(ValueError) as context:
                merge_recordings([sorted1, sorted2])

            self.assertIn("does not match earlier recordings", str(context.exception))

        finally:
            os.remove(path1)
            os.remove(path2)

if __name__ == "__main__":
    unittest.main()
