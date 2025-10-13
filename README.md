# RPMD-for-Kubo-Transformed-Correlation-Function

It aims to replicate the results from this paper:
Ian R. Craig, David E. Manolopoulos; Quantum statistics and classical mechanics: Real time correlation functions from ring polymer molecular dynamics. J. Chem. Phys. 22 August 2004; 121 (8): 3368–3373. https://doi.org/10.1063/1.1777575

Here I calculated the (Kubo transformed) correlation function for harmonic, slightly anharmonic and completely anharmonic potentials using three methods:

- Exact quantum (by diagonalising the Hamiltonian matrix and evolving the eigenstates)
- Classical mechanics (propagating classical states with verlet algorithm)
- Ring polymer molecular dynamics (Metropolis sampling + verlet propagator)

The folder named RPMD_Correlation_MC is my initial attemp where I used a (very long) Metropolis Monte Carlo pre-equilibration before each ring polymer sampling to achieve Boltzmann statistics. This is clearly suboptimal. Also I propagated the ring polymer dynamics using the naive primitive velocity verlet algorithm.

My second attempt is in RPMD_Correlation_Andersen, where the Boltzmann statistics is achieved by kicking the system with a Andersen thermostat every time after each sampling step. The random Boltzmann sampling of coordinates is naturally achieved by starting each simulation from the final position of the previous run. I also propagated the internal vibration of polymers exactly by transforming into the normal mode coordinates. This is much more accurate, and FFT algorithm can be used, which is fast.

Each folder has two versions: one with numba and one without numba. The numba version runs much faster, but the codes are less structured (ugly).
