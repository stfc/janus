# CC_HDNNP

CC_HDNNP is a collection of utility scripts for Machine Learnt Potential (MLP) workflows using SLURM for job submission, and interfacing with the following software:
  - CP2K and Quantum Espresso to generate training and testing data
  - N2P2 for the ML
  - LAMMPS for using the MLP as a classical forcefield
  - ASE for file handling and atomic representations
  - AML for validation testing

Overall workflow is as follows:
  1. Generate dataset using either CP2K or Quantum Espresso
  2. Run training using N2P2
  3. Using the trained network for LAMMPS simulations, including validation
  4. Use active learning to expand the starting dataset
  5. Remove undesired points from the dataset, or symmetry functions from the atomic representations, whilst keeping it representative of the parameter space.

For the functional details, see the notebooks in `examples/example_directory/scripts`.
