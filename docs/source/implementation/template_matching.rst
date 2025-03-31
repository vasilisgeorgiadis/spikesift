.. _template_matching:

Template Matching
=================

After :ref:`template_formation`, SpikeSift scans for **similar spikes** throughout the current segment.

Why recover more spikes after clustering?
-----------------------------------------

Initial detection uses a **strict threshold** to ensure reliability. 
But that means many valid spikes --- especially those close to the detection threshold --- may go **undetected**.

Does this mean scanning the entire signal?
------------------------------------------

No. SpikeSift avoids full convolutions. 
It begins with a shortlist of likely spike locations (**local minima**) and then checks each one for **similarity** to the current template.

How is similarity evaluated efficiently?
----------------------------------------

Most candidates are filtered out with a fast **amplitude-based test** across the five nearest channels. 
This lightweight check eliminates the majority of false detections.

Why not rely on a fixed threshold?
----------------------------------

Because a fixed cutoff rarely separates spikes from noise **across all conditions**. 
Instead, SpikeSift uses a :ref:`binary_splitting_clustering` step with the template as reference to isolate a clean unit.

What if clustering fails?
-------------------------

If the new cluster is **too small** (i.e., fewer than :math:`N_{\min}` (``min_spikes_per_cluster``) spikes) or **too weak** (below the detection threshold), it's discarded. 
The respective **reference channel** used in :ref:`template_formation` is also skipped in future passes to avoid unnecessary computation.

What happens to accepted spikes?
--------------------------------

Their average waveform is **subtracted** at each occurrence. 
This removes interference from strong units and helps uncover **weaker or overlapping spikes** in later iterations.
