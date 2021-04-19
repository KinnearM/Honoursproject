# Honoursproject
Python code pertaining to my GW lab project.

So far contains 5 files:

extradim.py contains leakageparam class object which has all the tools you could need (I could think of) to infer the number of spacetime dimensions from GW events.
Requires numpy, scipy, astropy, emcee, corner, matplotlib and cosmo.

cosmo.py contains lots of neat cosmology functions and is required by extradim.

dVddL.txt is a uniform in comoving-volume distance prior assuming a Planck 2015 cosmology. Required if your distance posterior had this prior.

demo.ipynb is a python notebook demonstrating the awesome power of extradim

distpostGW19.txt is a text file containg distance posterior samples used in the demo.

