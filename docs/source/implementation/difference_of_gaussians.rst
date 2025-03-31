.. _difference_of_gaussians:

Difference of Gaussians
=======================

SpikeSift applies a **Difference-of-Gaussians (DoG)** filter to isolate the frequency band where spikes are most prominent.  
Because the rest of the pipeline is highly optimized, filtering accounts for nearly **one-third of total runtime** --- making an efficient implementation essential.

Why Gaussians?
--------------

Gaussian filters apply a **smooth weighting**, emphasizing nearby samples while gently suppressing distant ones.  
This avoids **artifacts and sharp transitions** introduced by filters with abrupt cutoffs.

Why take the difference of two Gaussians?
-----------------------------------------

Subtracting two Gaussians produces a clean **bandpass effect**:  

- The **narrow** Gaussian removes **high-frequency noise**  
- The **wide** Gaussian eliminates **slow fluctuations** like local field potentials

Why is this more efficient than standard filters?
-------------------------------------------------

Each Gaussian is approximated using four **recursive moving averages** (box filters).  
This avoids expensive operations like **convolutions** or **FFTs**, while still preserving signal fidelity.