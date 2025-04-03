.. _implementation:

Implementation
==============

SpikeSift performs spike sorting in three main stages:

1. **Filtering and Segmentation:**  
   The raw signal is first filtered using a :ref:`difference_of_gaussians` filter to isolate **spike-relevant frequencies**.  
   It is then split into segments via :ref:`adaptive_segmentation`,
   which monitors **spike amplitude fluctuations** across channels to detect **electrode drift**.

2. **Iterative Sorting:**  
   Within each segment, spikes are sorted through an :ref:`iterative_sorting` process.  
   Neurons are resolved **one at a time** via :ref:`template_formation` and :ref:`template_matching`,
   with both steps relying on :ref:`binary_splitting_clustering` to isolate spikes from a **single unit**.

3. **Segment Merging:**  
   Finally, clusters are **aligned across segments** via :ref:`segment_merging`.  
   Since each neuron typically appears as a **single, well-isolated cluster per segment**, 
   merging reduces to a simple, reliable **one-to-one matching problem** --- minimizing the risk of cluster fragmentation.

Each stage is designed for **clarity**, **computational efficiency**, and **robustness to drift** --- without requiring precise drift correction or hardware acceleration.
For a full description of the algorithm, see `SpikeSift: A Computationally Efficient and Drift-Resilient Spike Sorting Algorithm <https://arxiv.org/abs/2504.01604>`__.

.. toctree::
   :maxdepth: 1
   :hidden:

   implementation/difference_of_gaussians
   implementation/adaptive_segmentation
   implementation/iterative_sorting
   implementation/segment_merging
