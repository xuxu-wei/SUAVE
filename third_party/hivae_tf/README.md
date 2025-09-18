# hivae2

This repository contains a modular wrapper of the "Heterogeneous Incomplete Variational Autoencoder model (HI-VAE)".  The previous implementation branch (v1) has been used in: 

Karwath et al., Redefining β-blocker response in heart failure patients with sinus rhythm and atrial fibrillation: a machine learning cluster analysis, Lancet, 2021 [paper](https://doi.org/10.1016/S0140-6736(21)01638-X). 

This modular wrapper was written by Andreas Karwath (a.karwath@bham.ac.uk) and Fathy Shalaby.

The original coding was done by Alfredo Nazabal (anazabal@turing.ac.uk) et al. written in Python and details can be found in this [paper](https://doi.org/10.1016/j.patcog.2020.107501). The original code can be found here: https://github.com/probabilistic-learning/HI-VAE

This is an extenstion of implementations as easy to use Python library, upgraded for tensorflow2.

Please cite both papers if you should use this code/library for your own research.


## Examples

See examples directory for usage

## Contact

* **For questions regarding algorithm --> Alfredo Nazabal**: anazabal@turing.ac.uk
* **For bugs or suggestion regarding this code --> Andreas Karwath**: a.karwath@bham.ac.uk

## Comments

This version requires tf2 (please not because of issues with installing tf2 on an Apple silicon, this is not specified in the setup.py as a requirement). For apple silcone users, please follow : https://developer.apple.com/metal/tensorflow-plugin/

