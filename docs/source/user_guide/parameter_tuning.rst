.. _parameter_tuning:

Parameter Tuning
================

SpikeSift exposes a compact set of **interpretable parameters** that control key trade-offs in spike sorting.  
All parameters are optional. The default values work robustly in most use cases.

Controlling Sensitivity vs. False Positives
-------------------------------------------

The ``detection_sensitivity`` parameter (denoted :math:`\kappa`) determines the **detection threshold**  
as a multiple of the median absolute deviation (MAD) on each channel.

.. code-block:: python

   detection_sensitivity = 10

- Lower values (e.g., ``6-8``) detect weaker spikes but may increase false positives  
- Higher values (e.g., ``12-14``) suppress noise but may miss low-amplitude events  

Controlling Cluster Merging vs. Splitting
-----------------------------------------

The ``merging_threshold`` parameter (denoted :math:`\lambda`) controls **how similar**  
two waveform profiles must be to be considered from the same neuron.

.. code-block:: python

   merging_threshold = 0.4

- Higher values (e.g., ``0.5-0.6``) merge more clusters (less conservative)  
- Lower values (e.g., ``0.2-0.3``) preserve more distinct clusters (more conservative)  

Including Rare Events
---------------------

The ``min_spikes_per_cluster`` parameter (denoted :math:`N_{\min}`) sets  
the **minimum number of spikes** required for a cluster to be retained.

.. code-block:: python

   min_spikes_per_cluster = 5

- Lower values (e.g., ``2-3``) allow detection of low-firing neurons  
- Higher values (e.g., ``10+``) improve cluster stability and precision  

Drift Alignment Resolution
--------------------------

The ``max_drift`` parameter (denoted :math:`D_{\max}`) defines the **maximum displacement (in micrometers)** 
allowed when aligning clusters across segments.

.. code-block:: python

   max_drift = 30  # micrometers

- Higher values (e.g., ``50-100``) allow alignment across larger probe displacements  
- Lower values (e.g., ``10-20``) restrict alignment to spatially consistent waveforms  

Minimum Segment Duration
------------------------

The ``min_segment_length`` parameter (denoted :math:`L_{\min}`) sets the minimum **duration (in seconds)**  
for each processing segment.

.. code-block:: python

   min_segment_length = 10  # seconds

- Smaller values (e.g., ``5``) improve drift tracking through finer segmentation  
- Larger values (e.g., ``20``) help detect rare events in stable recordings  

Controlling Detection Polarity and Scaling
------------------------------------------

The ``detection_polarity`` parameter is a scalar multiplier applied to the signal **during filtering**.  
It determines the direction of spikes to detect and optionally rescales the data for amplitude consistency.

.. code-block:: python

   detection_polarity = -1.0

- Use ``-1.0`` to detect negative-going spikes (default)
- Use ``+1.0`` to detect positive-going spikes
- Use a custom value  (e.g., ``-1.0 / a``) to undo scaling if the recording was linearly transformed

When Should You Change These?
-----------------------------

You may consider tuning these parameters if you need to:

- Increase sensitivity without over-detecting noise  
- Detect low-firing or low-amplitude spikes  
- Separate neurons with similar waveforms  
- Align spikes across faster or larger drift events  

For algorithmic context, see the :ref:`implementation`.