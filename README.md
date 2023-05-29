# SweeTile
This repository contains all the code required for analysis, imeplementation, and evaluation in "SweeTile: Efficient Tiling for 360° Video Streaming on Light-Weight VR Devices".

## Directory
### src/coverage
* ``sweet_all_area.py`` computes the coverage of 144 sweet spots over the visual sphere.
* Each sweet spot covers a square area with side length of $60^\circ-\alpha$.

### src/solver
* ``solver.py`` acquires all the tiles with high and mid quality given any possible viewpoint.

### src/main
* Three notebooks ``sweetile.ipynb``, ``sweetile_6x3.ipynb``, and ``sweetile_tbra.ipynb`` implement the proposed and benchmark algorithms.

### trace/*
* All the required bandwidth traces and viewport predictions are organized in this folder. 
