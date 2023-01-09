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

## Installation instructions

### AML

- Download the AML repository:
  - `git clone https://github.com/MarsalekGroup/aml`
  - `cd aml`
  - Create a `pyproject.toml` file e.g.:
    - ```
      [tool.poetry]
      name = "aml"
      version = "1.0.0"
      description = "This is a Python package to automatically build the reference set for the training of Neural Network Potentials (NNPs), and eventually other machine-learned potentials, in an automated, data-driven fashion. For that purpose, a large set of reference configurations sampled in a physically meaningful way (typically with molecular dynamics) is filtered and the most important points for the representation of the Potential Energy Surface (PES) are identified. This is done by using a set of NNPs, called a committee, for error estimates of individual configurations. By iteratively adding the points with the largest error in the energy/force prediction, the reference set is progressively extended and optimized."
      authors = ["O Marsalek"]
      [tool.poetry.dependencies]
      mdtraj = "^1.9.6"
      ```

### CC_HDNNP

- Download the CC_HDNNP repository:
  - `cd ..`
  - `git clone https://github.com/stfc/CC_HDNNP`
  - `cd CC_HDNNP`
- Install poetry:
  - https://python-poetry.org/docs/
  - e.g. `curl -sSL https://install.python-poetry.org | python3 -`
- Install dependencies
  - (Optional) Create a virtual environment
  - `poetry install`
    - If you are already in a virtual environment, packages will be installed in this environment
    - Otherwise, a new virtual environment will be created, which you must then activate
- Build and install:
  - `python3 setup.py build && python3 setup.py install`

### n2p2

- Download the n2p2 repository:
  - `cd ..`
  - `git clone https://github.com/CompPhysVienna/n2p2`
- Build:
  - `cd n2p2/src`
  - `make -j 4`

### LAMMPS

- Download the LAMMPS repository:
  - `cd ../..`
  - `git clone https://github.com/lammps/lammps`
- Build:
  - `mkdir build; cd build`
  - `cmake ../cmake -D PGK_ML-HDNNP=yes -D N2P2_DIR=/path/to/n2p2/`
  - `cmake --build .`

