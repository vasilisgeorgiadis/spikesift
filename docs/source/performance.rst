.. _performance:

Performance
===========

SpikeSift achieves **significant speed improvements** over traditional spike sorters ---  
outperforming even **GPU-accelerated methods** like Kilosort --- while running entirely on a **single CPU core**.

Performance Highlights
----------------------

SpikeSift maintains consistently **high performance** across a wide range of probe sizes, recording lengths, and experimental conditions.  

It is especially effective for **high-throughput** pipelines, **exploratory** workflows, and **real-time** applications where spike sorting must outpace data acquisition.

Some key performance characteristics include:

- Over **20× faster** than Kilosort on high-density datasets  
- Up to **300× faster** than Kilosort when both run on a single CPU core  
- **Real-time** sorting of thousands of channels using a **single CPU core**  
- Efficient even on **short recordings** (under 10 seconds)  
- Robust to both **abrupt and continuous electrode drift**  
- Supports **parallel and progressive sorting** with minimal overhead  

SpikeSift is so efficient that complete spike sorting often finishes  
**faster than simply copying or saving the data** using NumPy I/O operations.  
This makes it practical even in workflows where I/O or preprocessing would normally dominate runtime.

Run a Quick Benchmark
---------------------

To measure runtime on your system, wrap the sorting call with a timer:

.. code-block:: python

   import time
   from spikesift import perform_spike_sorting

   start = time.time()
   result = perform_spike_sorting(recording)
   spikes = result.all_spikes()
   end = time.time()

   print(f"Spike sorting completed in {end - start:.2f} seconds.")

This gives a quick end-to-end runtime estimate.  
For meaningful results, use a realistic recording with appropriate duration and number of channels.

.. note::

   This is intended as a basic diagnostic.  
   For detailed benchmarks and comparisons, see the upcoming SpikeSift paper:

   - `SpikeSift: A Computationally Efficient and Drift-Resilient Spike Sorting Algorithm <https://iopscience.iop.org/article/10.1088/1741-2552/adee48>`__

Why Is SpikeSift So Fast?
-------------------------

SpikeSift's speed comes from its **algorithmic design**, not just low-level optimization:

- Uses **lightweight template matching**, avoiding full waveform convolutions  
- Applies an **iterative detect-and-subtract process**, isolating one neuron at a time  
- Focuses on **local neighborhoods**, avoiding expensive global clustering  
- Loads each segment into memory only **once**, thanks to adaptive segmentation  
- Implements all performance-critical steps in **Cython**, maximizing CPU efficiency

This makes SpikeSift suitable for dense probes, short recordings, and high-volume datasets --- 
even in **real-time** applications, using **a single CPU core** and **minimal memory**.
