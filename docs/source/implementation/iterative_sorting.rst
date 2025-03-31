.. _iterative_sorting:

Iterative Sorting
=================

SpikeSift performs a **loop** of :ref:`template_formation` and :ref:`template_matching`,  
using :ref:`binary_splitting_clustering` to isolate spikes from **one neuron at a time**.

Why isolate one neuron at a time?
---------------------------------

It makes the problem **simpler and more robust**.  
Clustering all spikes at once is difficult --- especially when waveforms overlap or vary in amplitude.  
By removing **strong units** early, SpikeSift clears the way for **weaker or overlapping spikes** that would otherwise be missed.

Why use a template matching approach?
-------------------------------------

Without a clear **template waveform**, it's hard to recover all spikes from a neuron without also capturing noise or spikes from other neurons.
SpikeSift starts with a small, reliable cluster to build a clean template, used to detect additional matches --- improving **sensitivity** without losing **precision**.

When does the loop stop?
------------------------

The loop ends when no more units can be isolated --- either because the remaining spikes are **too weak** or
**too inconsistent** to form a reliable cluster.

.. toctree::
   :hidden:

   template_formation
   binary_splitting_clustering
   template_matching
