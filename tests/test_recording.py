import numpy as np
import tempfile
import os
from spikesift import Recording
import unittest

class TestRecording(unittest.TestCase):

    def test_recording_read_basic(self):
        num_channels = 4
        num_samples = 100
        data = (np.arange(num_samples * num_channels) % 256).astype("int16")
        binary_data = data.tobytes()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(binary_data)
            path = f.name

        try:
            
            geometry = np.zeros((num_channels, 2))
            rec = Recording(
                binary_file=path,
                data_type="int16",
                probe_geometry=geometry,
                sampling_frequency=20000,
            )

            self.assertEqual(rec.num_samples, num_samples)
            self.assertEqual(rec.recording_channels, num_channels)
            self.assertEqual(rec.bytes_per_sample, num_channels * np.dtype("int16").itemsize)
            self.assertAlmostEqual(rec.samples_per_ms, 20)
            read_data = rec.read(start=0, num_samples=num_samples)
            self.assertEqual(read_data.shape, (num_samples, num_channels))
            self.assertTrue(np.array_equal(read_data.ravel(), data))

        finally:

            os.remove(path)

    def test_recording_offset_and_header(self):
        num_channels = 4
        num_samples = 50
        data = np.arange(num_samples * num_channels, dtype="int16")

        header = b'HEADER1234'
        offset_samples = 20
        padded_data = np.concatenate([
            np.zeros(offset_samples * num_channels, dtype="int16"),
            data
        ]).tobytes()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(header)
            f.write(padded_data)
            path = f.name

        try:
    
            geometry = np.zeros((num_channels, 2))
            rec = Recording(
                binary_file=path,
                data_type="int16",
                probe_geometry=geometry,
                sampling_frequency=10000,
                header=len(header),
                sample_offset=offset_samples,
                num_samples=num_samples
            )

            read_data = rec.read(start=0, num_samples=num_samples)
            self.assertEqual(read_data.shape, (num_samples, num_channels))
            self.assertTrue(np.array_equal(read_data.ravel(), data))

        finally:

            os.remove(path)

    def test_invalid_data_type_raises(self):
        geometry = np.zeros((4, 2))
        with self.assertRaises(ValueError):
            Recording(
                binary_file="nonexistent.dat",
                data_type="complex128",
                probe_geometry=geometry,
                sampling_frequency=20000,
            )

if __name__ == '__main__':
    unittest.main()