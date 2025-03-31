.. _template_formation:

Template Formation
==================

Each iteration begins by isolating a clean **template waveform** --- a representative spike shape 
used in :ref:`template_matching` to recover all spikes from the same neuron.

How are candidate spikes detected?
----------------------------------

SpikeSift scans for **local minima** on the channel with the **strongest cumulative spike amplitude** above threshold.  
This **reference channel** favors consistent activity over isolated outliers.

How is the detection threshold set?
-----------------------------------

The threshold is proportional to the **median absolute deviation (MAD)**.  
The detection sensitivity parameter :math:`\kappa` (``detection_sensitivity``) controls how strict the threshold is.

How are spike waveforms extracted?
----------------------------------

Each detected spike is extracted as a **2-ms** waveform from the **five nearest electrodes**.  
This captures spatial structure while keeping the dimensionality low.

What happens to the extracted waveforms?
----------------------------------------
                
They are passed to :ref:`binary_splitting_clustering`, which isolates a single, well-defined cluster.  
The **mean waveform** of that cluster becomes the template for this iteration.
