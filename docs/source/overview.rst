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

SpikeSift is designed for speed, robustness, and flexibility:

- **Extremely fast** --- processes thousands of channels in real time 
- **Resilient to drift** --- handles both slow drift and sudden shifts  
- **Modular** --- sort in parallel, split/merge segments, track transients  
- **Easy to use** --- most datasets work out of the box  
- **Progressive-compatible** --- supports merging results across sessions or files  
- **Accurate on short recordings** --- retains spike sorting quality even when data is limited

These features make it ideal for real-time analysis, high-throughput pipelines, and exploratory sorting across large datasets.

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
    
    - (link coming soon)