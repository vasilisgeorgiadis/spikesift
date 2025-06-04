.. _adaptive_segmentation:

Adaptive Segmentation
=====================

Before :ref:`iterative_sorting`, SpikeSift divides the recording into **segments** where electrode drift is minimal.  
This improves clustering accuracy and reduces the risk of misclassifying spikes due to waveform changes over time.

How are segment boundaries selected?
------------------------------------

SpikeSift monitors how **spike amplitudes** evolve across all channels.  
Boundaries are placed where **multiple channels** show **abrupt, simultaneous changes** --- a strong indicator of drift.

Why not use fixed-length windows?
---------------------------------

Because drift is **unpredictable**.  
By placing boundaries **where changes are most pronounced**, SpikeSift adapts to the recording's actual dynamics --- without relying on **predefined time intervals**.

How is over-segmentation avoided?
---------------------------------

Segments are enforced to be at least :math:`L_{\min}` (``min_segment_length``) seconds long.  
This ensures that each segment contains enough spikes for **reliable clustering** and avoids **excessive fragmentation**.

Does this require loading the entire recording?
-----------------------------------------------

No. SpikeSift uses a cyclic buffer that processes data in memory-efficient batches.
Only a few seconds of data are loaded at a time, so the full recording does not need to fit in RAM.
