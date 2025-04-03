.. _overview:

Overview
========

SpikeSift is a high-performance spike sorting algorithm designed for high-density extracellular recordings.  
It delivers **state-of-the-art** accuracy while running in **real time on a single CPU core**.

How Does SpikeSift Work?
------------------------

SpikeSift extracts and tracks individual neuron activity from raw extracellular recordings.  
It includes:

- Filtering and segmenting the signal based on drift-aware heuristics  
- Iteratively detecting and clustering spikes within each segment  
- Merging matching clusters across segments to produce globally aligned spike trains  

The result is a drift-corrected, neuron-by-neuron reconstruction of spiking activity across time.

Why Use SpikeSift?
------------------

SpikeSift is built for speed, robustness, and clean integration into any workflow:

- **Extremely fast** --- sorts thousands of channels in real time on a single CPU core
- **Drift-resilient** --- handles both gradual and abrupt electrode drift
- **Clean and non-intrusive** --- no data copying, no file modifications, no clutter
- **Modular** --- sort in parallel, split or merge segments, track transients
- **Drop-in ready** --- works out of the box on most datasets
- **Session-aware** --- merge across files, append sessions, or sort progressively
- **Reliable on short recordings** --- maintains accuracy even with limited data

These features make SpikeSift ideal for real-time pipelines, high-throughput labs, and large-scale sorting tasks --- even on resource-constrained systems.

What the Documentation Covers
-----------------------------

The rest of this documentation includes:

- :ref:`installation` --- how to install and get started  
- :ref:`user_guide` --- how to sort data and interpret the results  
- :ref:`example_usage` --- real-world workflows and advanced use cases  
- :ref:`performance` --- benchmarks and key efficiency advantages  
- :ref:`implementation` --- algorithmic insights and design principles  
- :ref:`api_reference` --- complete API for all user-facing functions  

For a quick start, see the :ref:`user_guide`.  
To explore practical workflows, head to :ref:`example_usage`.

.. note::

    SpikeSift is under active development and continues to improve in accuracy and flexibility.  
    For more details or citations, see the upcoming preprint:
    
    - `SpikeSift: A Computationally Efficient and Drift-Resilient Spike Sorting Algorithm <https://arxiv.org/abs/2504.01604>`__
