.. _binary_splitting_clustering:

Binary-Splitting Clustering
===========================

SpikeSift isolates spikes from a single neuron by recursively applying **binary splits** to the detected waveforms.  
This step filters out **noise** and spikes from other neurons with **similar waveforms**.

How are waveforms split?
------------------------

Waveforms are projected onto their **principal axis of variance** using power iteration,  
then separated into two candidate groups by a fast **1D nearest-neighbor chain** clustering algorithm.

Why use binary splits?
----------------------

Binary splits are **simple, efficient, and adaptive**.  
Instead of choosing the number of clusters in advance, SpikeSift **refines the waveforms iteratively**, splitting only when needed.

How does SpikeSift decide when to stop?
---------------------------------------

After each split, the two candidate clusters are compared using their **spatial waveform structure**.  
If they are sufficiently similar --- measured by their **relative Euclidean distance**---they are **merged**.  
This decision is governed by a **similarity threshold** :math:`\lambda` (``merging_threshold``).

How is spatial waveform structure captured?
-------------------------------------------

Each cluster's average waveform is converted into a compact **inter-channel difference vector**,  
which encodes the **maximum voltage difference** between every pair of electrodes.

Which cluster is retained?
--------------------------

In :ref:`template_formation`, SpikeSift keeps the cluster with the **larger average amplitude**.  
In :ref:`template_matching`, it keeps the one that **best matches the current template**.
