# RPMD-for-Kubo-Transformed-Correlation-Function

It aims to replicate the results from this paper:
Ian R. Craig, David E. Manolopoulos; Quantum statistics and classical mechanics: Real time correlation functions from ring polymer molecular dynamics. J. Chem. Phys. 22 August 2004; 121 (8): 3368–3373. https://doi.org/10.1063/1.1777575

Here I calculated the (Kubo transformed) correlation function for harmonic, slightly anharmonic and completely anharmonic potentials using three methods:

- Exact quantum (by diagonalising the Hamiltonian matrix and evolving the eigenstates)
- Classical mechanics (propagating classical states with verlet algorithm)
- Ring polymer molecular dynamics (Metropolis sampling + verlet propagator)

It uses an Andersen thermostat which kicks the system after each run is completed, and a Boltzmann statistics is naturally achieved by using the end configuration of the previous run as the starting configuration of the next run. It propagates the dynamics using a verlet propagator, which is modified to transform the real coordinates to the normal mode coordinates so the internal vibrations of the polymer can be propagated exactly.

The notebook with numba runs much faster, but the one without numba has more captions and the codes are better structured.

I also included an interactive python script which allows you to see the sampling happens in real time. This is much less efficient (because I need to delibrately add delay to the verlet propagator so the movement of the polymer is visible) but quite entertaining to look at!

The scripts in the MC folder is my initial attempt. It uses Metropolis Monte Carlo so it's very inefficient if the number of beads is large. No need to look at it! It's there just for reference (and to remind me how poor my coding skill was).
