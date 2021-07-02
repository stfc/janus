# CC_HDNNP

CC_HDNNP is a collection of utility scripts for running Machine Learnt Potential (MLP) workflows on SCARF using the following software:
  - CP2K for DFT to get training and testing data
  - N2P2 for the ML
  - LAMMPS for using the MLP as a classical forcefield
  - ASE for some file conversions

Overall workflow is as follows:
  1. Generate atomic configurations for system
  2. Write CP2K files (input and batch scripts) for a number of configurations and cutoff parameters
  3. Run CP2K
  4. Choose cutoff parameters based on the CP2K results (and if needed re-run CP2K with a larger number of frames)
  5. Write N2P2 input from CP2K output
  6. Scale and prune symmetry functions
  7. Train network
  8. Write output for LAMMPS from N2P2 output
  9. Run LAMMPS

For the functional details, see the examples folder.