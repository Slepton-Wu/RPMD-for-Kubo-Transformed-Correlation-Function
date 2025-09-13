# RPMD-for-Kubo-Transformed-Correlation-Function

Here I calculated the (Kubo transformed) correlation function for harmonic, slightly anharmonic and completely anharmonic potentials using three methods:
- Exact quantum (by diagonalising the Hamiltonian matrix and evolving the eigenstates)
- Classical mechanics (propagating classical states with verlet algorithm)
- Ring polymer molecular dynamics (Metropolis sampling + verlet propagator)

It aims to replicate the result from this paper:
Ian R. Craig, David E. Manolopoulos; Quantum statistics and classical mechanics: Real time correlation functions from ring polymer molecular dynamics. J. Chem. Phys. 22 August 2004; 121 (8): 3368–3373. https://doi.org/10.1063/1.1777575
