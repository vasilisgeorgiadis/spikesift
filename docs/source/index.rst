.. SpikeSift documentation master file

Welcome to SpikeSift
====================

SpikeSift is a **fast**, **drift-resilient** spike sorting algorithm for high-density extracellular recordings.  
It runs in **real time** on a **single CPU core** and can be parallelized across segments for large-scale processing.  
Designed for **speed**, **modularity**, and **robustness**, it supports adaptive segmentation, segment merging, 
and progressive recording analysis with minimal tuning.

.. toctree::
   :maxdepth: 2

   overview
   installation
   user_guide
   example_usage
   performance
   implementation
   api_reference

If you're new to SpikeSift, start with the :ref:`overview`, then follow the :ref:`user_guide` to run your first sort.
