.. _segment_merging:

Segment Merging
===============

After :ref:`iterative_sorting`, **clusters are aligned** across segments to preserve **neuron identity over time**.  
This step is **very efficient** --- typically less than **1% of total processing time**.

How is each cluster represented?
--------------------------------

Each cluster is reduced to a compact **amplitude vector** --- one value per channel --- capturing its average waveform shape.  
This representation is **fast** to compare and **robust** to minor distortions.

How is drift handled?
---------------------

SpikeSift simulates **small vertical shifts** along the probe axis --- up to a user-defined maximum :math:`D_{\max}` (``max_drift``).  
Each shift adjusts amplitude vectors using **linear interpolation**, approximating how waveforms change  
as a neuron moves relative to the probe.

How are clusters matched?
-------------------------

For each candidate shift, SpikeSift finds the best **one-to-one matching** between clusters in adjacent segments.  
The shift with the **lowest total assignment cost** is selected.
