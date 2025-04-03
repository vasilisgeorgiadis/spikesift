# SpikeSift

**SpikeSift** is a **fast**, **drift-resilient** spike sorting algorithm for high-density extracellular recordings.
It delivers **accurate**, **real-time** spike sorting from raw binary data using only a single CPU core.

---

## Features

- 🔥 **Real-time performance** on thousands of channels
- 🧠 **Drift-aware segmentation** and robust merging 
- ⚡ **Parallelizable** across segments  
- 🧩 **Modular design** — sort, merge, split, and compare segments 
- 🎯 **Minimal parameter tuning**, even for short recordings
- 💾 Reads data directly from disk — no need to convert or preprocess
- 🧽 Clean and non-intrusive — does not modify your data or create intermediate files

---

## Installation

Install with `pip`:

```bash
pip install spikesift
```

Or install from source:

```bash
git clone https://github.com/vasilisgeorgiadis/spikesift.git
cd spikesift
pip install -e .
```

---

## Quickstart

```python
from spikesift import Recording, perform_spike_sorting

# Define probe layout (example)
import numpy as np
probe = np.load("probe.npy")

# Load raw data
recording = Recording(
    binary_file="recording.bin",
    data_type="int16",
    probe_geometry=probe,
    sampling_frequency=30000
)

# Run sorting
result = perform_spike_sorting(recording)

# Access spike times
for cid in result.cluster_ids():
    spikes = result.spikes(cid)
    print(f"Cluster {cid}: {len(spikes)} spikes")
```

For more examples, see the [User Guide](https://spikesift.readthedocs.io/en/latest/user_guide.html) or [Example Usage](https://spikesift.readthedocs.io/en/latest/example_usage.html).

---

## Documentation

Full documentation is available at:

**[https://spikesift.readthedocs.io/en/latest/index.html](https://spikesift.readthedocs.io/en/latest/index.html)**

---

## Performance

SpikeSift is over 20× faster than GPU-based sorters like Kilosort, and up to 300× faster when all run on a single CPU core.
It handles thousands of channels, fragmented recordings, and real-time pipelines with ease.

---

## Citing SpikeSift

**[SpikeSift: A Computationally Efficient and Drift-Resilient Spike Sorting Algorithm](https://arxiv.org/abs/2504.01604)**

---

## License

[MIT](LICENSE) © 2025 Vasileios Georgiadis
