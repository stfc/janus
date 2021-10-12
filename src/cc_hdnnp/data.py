"""
Read data from file and convert it to other formats so it can be used as input
for the next stage of the pipeline.

ASE uses units of:
    Length: Angstrom
    Energy: eV
    Force:  eV / Ang

cp2k uses units of:
    Length: Angstrom
    Energy: Ha
    Force:  Ha / Bohr
"""

from os import mkdir
from os.path import exists, isdir, isfile, join as join_paths
import re
from shutil import copy
import time
from typing import Dict, List, Literal, Tuple, Union

from ase.atoms import Atoms
from ase.io import read, write
from ase.units import create_units
import numpy as np

from cc_hdnnp.data_operations import check_structure
from cc_hdnnp.file_operations import (
    format_template_file,
    read_atomenv,
    read_data_file,
    read_nn_settings,
    read_scaling,
    remove_data,
)
from cc_hdnnp.sfparamgen import SymFuncParamGenerator
from cc_hdnnp.structure import AllStructures, Structure


class Data:
    """
    Holds information relevant to reading and writing data from and to file.

    Parameters
    ----------
    structures : AllStructures
        Structures that are being described by the network.
    main_directory : str
        Path to the directory that contains all the other sub-directories relevant to the
        network. Other file paths specified should be relative to this, excluding bin
        directories for external software.
    lammps_executable : str
        Path of the LAMMPS executable.
    n2p2_bin : str
        Path to the n2p2 bin directory.
    scripts_sub_directory : str, optional
        Path for the directory to read/write scripts from/to, relative to the
        `main_directory`. Default is "scripts".
    n2p2_sub_directory : str, optional
        Path for the directory to read/write n2p2 files from/to, relative to the
        `main_directory`. Default is "n2p2".
    active_learning_sub_directory : str, optional
        Path for the directory to read/write active learning files from/to, relative to the
        `main_directory`. Default is "active_learning".
    lammps_sub_directory : str, optional
        Path for the directory to read/write LAMMPS files from/to, relative to the
        `main_directory`. Default is "lammps".
    """

    def __init__(
        self,
        structures: AllStructures,
        main_directory: str,
        lammps_executable: str,
        n2p2_bin: str,
        scripts_sub_directory: str = "scripts",
        n2p2_sub_directory: str = "n2p2",
        active_learning_sub_directory: str = "active_learning",
        lammps_sub_directory: str = "lammps",
    ):
        self.units = create_units("2014")
        self.units["au"] = self.units["Bohr"]
        self.all_structures = structures
        # TODO convert to getters
        self.structure_names = structures.structure_dict.keys()
        self.elements = structures.element_list
        self.main_directory = main_directory
        self.lammps_executable = lammps_executable
        self.n2p2_bin = n2p2_bin
        self.scripts_directory = join_paths(main_directory, scripts_sub_directory)
        self.n2p2_directory = join_paths(main_directory, n2p2_sub_directory)
        self.active_learning_directory = join_paths(
            main_directory, active_learning_sub_directory
        )
        self.lammps_directory = join_paths(main_directory, lammps_sub_directory)

        self.trajectory = None

    def _min_n_config(self, n_provided: int):
        """
        Utility function to ensure we don't attempt to read/write more frames
        than are available.

        Parameters
        ----------
        n_provided : int
            Number of frames specified by the user.

        Returns
        -------
        int
            The minimum of `n_provided` and the length of `self.trajectory`
            they exist. Returns `0` if both are `None`.
        """
        if n_provided is not None and self.trajectory is not None:
            n_config = min(n_provided, len(self.trajectory))
        elif n_provided is not None:
            n_config = n_provided
        elif self.trajectory is not None:
            n_config = len(self.trajectory)
        else:
            n_config = 0

        return n_config

    def read_trajectory(
        self, file_trajectory: str, format_in: str = "dlp-history", unit_in: str = "Ang"
    ):
        """
        Reads `file_trajectory` and stores the trajectory. By default length
        units of Ang are assumed, so if this is not the case then `unit_in`
        should be specified to allow conversion. The resulting
        `self.trajectory` will be in 'Ang'.

        Parameters
        ----------
        file_trajectory : str
            File containing a trajectory of atom configurations, relative to the main directory.
        format_in : str, optional
            File format for the ASE reader. Default is 'dlp-history'.
        unit_in: str, optional
            Length unit for the trajectory. Default is 'Ang'.
        """
        trajectory = read(
            join_paths(self.main_directory, file_trajectory),
            format=format_in,
            index=":",
        )

        if unit_in != "Ang":
            for frame in trajectory:
                cell = frame.get_cell()
                positions = frame.get_positions()
                frame.set_cell(cell * self.units[unit_in])
                frame.set_positions(positions * self.units[unit_in])

        self.trajectory = trajectory

    def convert_active_learning_to_xyz(
        self,
        file_structure: str,
        file_xyz: str,
        unit_in: str = "Bohr",
        single_output: bool = False,
    ):
        """
        Reads `file_structure` from `self.active_learning_directory` and writes it as a series
        of xyz files. By default length units of Bohr are assumed, so if this is not the case
        then `unit_in` should be specified to allow conversion. The xyz files will be in Ang.

        Parameters
        ----------
        file_structure : str
            File containing a series of structures in n2p2 format, relative to
            `self.active_learning_directory`.
        file_xyz : str
            Formatable file name to write to, relative to `self.main_directory`.
        unit_in: str, optional
            Length unit for the trajectory. Default is 'Ang'.
        """
        xyz_path_split = join_paths(self.main_directory, file_xyz).split("/")
        xyz_directory = "/".join(xyz_path_split[:-1])
        if not isdir(xyz_directory):
            mkdir(xyz_directory)

        with open(join_paths(self.active_learning_directory, file_structure)) as f:
            lines = f.readlines()

        i = 0
        for line in lines:
            if line.strip() == "begin":
                atoms = []
                lattice = []
            elif line.split()[0] == "lattice":
                for j in (1, 2, 3):
                    lattice.append(str(float(line.split()[j]) * self.units[unit_in]))
            elif line.split()[0] == "atom":
                atom = [
                    line.split()[4],
                    str(float(line.split()[1]) * self.units[unit_in]),
                    str(float(line.split()[2]) * self.units[unit_in]),
                    str(float(line.split()[3]) * self.units[unit_in]),
                ]
                atoms.append(atom)
            elif line.strip() == "end":
                text = str(len(atoms))
                text += (
                    '\nLattice="{}" Properties=species:S:1:pos:R:3 pbc="T T T"'.format(
                        " ".join(lattice)
                    )
                )
                for atom in atoms:
                    text += "\n" + " ".join(atom)
                if single_output:
                    with open(join_paths(self.main_directory, file_xyz), "a") as f:
                        f.write(text + "\n")
                else:
                    with open(
                        join_paths(self.main_directory, file_xyz.format(i)), "w"
                    ) as f:
                        f.write(text)
                i += 1

    def remove_outliers(
        self,
        energy_threshold: Union[float, Tuple[float, float]],
        force_threshold: float,
        data_file_in: str = "input.data",
        data_file_out: str = "input.data",
        data_file_backup: str = "input.data",
        reference_file: str = "output.data",
    ):
        """
        Read `reference_file` for energy and force values, and in these and in `target_file`
        comment out the structures which have an energy or force above the specified
        threshold values.

        Practically, if "input.data" is the `reference_file`, then the `energy_threshold` and
        `force_threshold` should be in the same units as that file, i.e. physical units. If
        "output.data" is used, then normalised thresholds can be given (i.e. setting the
        theshold in multiples of the datasets standard deviations).

        Note that scaling, normalising and pruning should be redone after this.

        Parameters
        ----------
        energy_threshold : float or tuple of float
            Structures which lie outside the range of `energy_threshold` will be removed from
            the dataset. The units depend on `reference_file`.
        force_threshold : float
            Structures with a force component more than `force_threshold` will be removed from
            the dataset. The units depend on `reference_file`.
        data_file_in: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`,
            to read from. Default is "input.data".
        data_file_out: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`,
            to write to. Default is "input.data".
        data_file_backup: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`, to copy
            the original `data_file_in` to. Default is "input.data.minimum_separation_backup".
        reference_file: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`,
            to read the energy and force values from. Default is "input.data".
        """
        if isinstance(energy_threshold, float):
            energy_threshold = (-energy_threshold, energy_threshold)

        with open(join_paths(self.n2p2_directory, reference_file)) as f:
            lines = f.readlines()

        remove_indices = []
        remove = False
        atom_count = 0
        i = 0
        for line in lines:
            line_split = line.split()
            if line_split[0] == "atom":
                atom_count += 1
                force = np.array(
                    [
                        float(line_split[-3]),
                        float(line_split[-2]),
                        float(line_split[-1]),
                    ]
                )
                if any(abs(force) > force_threshold):
                    print(
                        "Structure {0} above threshold with a force of {1}".format(
                            i, force
                        )
                    )
                    remove = True
            elif line_split[0] == "energy":
                energy = float(line_split[-1]) / atom_count
                if energy < energy_threshold[0] or energy > energy_threshold[1]:
                    print(
                        "Structure {0} outside threshold with an energy of {1}".format(
                            i, energy
                        )
                    )
                    remove = True
            elif line_split[0] == "end":
                if remove:
                    remove_indices.append(i)
                remove = False
                atom_count = 0
                i += 1

        print("{0} outliers found: {1}".format(len(remove_indices), remove_indices))

        remove_data(
            remove_indices,
            join_paths(self.n2p2_directory, data_file_in),
            join_paths(self.n2p2_directory, data_file_out),
            join_paths(self.n2p2_directory, data_file_backup),
        )

    def write_xyz(self, file_xyz: str = "xyz/{}.xyz", unit_out: str = "Ang"):
        """
        Writes a loaded trajectory to file as a series of xyz files, optionally
        converting the units

        Parameters
        ----------
        file_xyz : str, optional
            Formatable file name to write the atomic co-ordinates to relative to
            `self.main_directory`. Will be formatted with the frame number, so should contain
            `'{}'` as part of the string. Default is 'xyz/{}.xyz'.
        unit_out: str, optional
            Length unit for the xyz files. Default is 'Ang'.

        Examples
        --------
        >>> from data import Data
        >>> d = Data('path/to/example_directory')
        >>> d.read_trajectory('example.history')
        >>> d.write_xyz('xyz/{}.xyz')
        """
        format_out = "extxyz"
        for i, frame in enumerate(self.trajectory):
            # Wrap coordinates to ensure all are positive and within the cell
            frame.wrap()
            if unit_out != "Ang":
                cell = frame.get_cell()
                positions = frame.get_positions()
                frame.set_cell(cell / self.units[unit_out])
                frame.set_positions(positions / self.units[unit_out])

            write(
                join_paths(self.main_directory, file_xyz.format(i)),
                frame,
                format=format_out,
                columns=["symbols", "positions"],
            )

    def scale_xyz(
        self,
        file_xyz_in: str = "xyz/{}.xyz",
        file_xyz_out: str = "xyz/{}.xyz",
        scale_factor: float = 0.05,
        randomise: bool = False,
        n_config: int = None,
    ):
        """
        Reads xyz files and rescales them by up to +-`scale_factor` before writing the
        scaled structure to file. Uniform distribution is used if `randomise` is `True`.

        Parameters
        ----------
        file_xyz_in : str, optional
            Formatable file name to read the atomic co-ordinates from, relative to
            `self.main_directory`. Will be formatted with the frame number, so should contain
            `'{}'` as part of the string. Default is 'xyz/{}.xyz'.
        file_xyz_out : str, optional
            Formatable file name to write the scaled atomic co-ordinates to, relative to
            `self.main_directory`. Will be formatted with the frame number, so should contain
            `'{}'` as part of the string. Default is 'xyz_scaled/{}.xyz'.
        scale_factor : float, optional
            The maximum relative change in length scale. Default is `0.05`.
        randomise: bool, optional
            Whether to choose the scale at random (within the maximum `scale_factor`) for each
            structure if `True`, or equally space the scaling across the range -+`scale_factor`
            for each structure sequentially. Default is `False`.
        """
        format = "extxyz"
        n_config = self._min_n_config(n_config)
        if randomise:
            scale = 1 + (2 * np.random.random(n_config) - 1) * scale_factor
        else:
            scale = np.linspace(1 - scale_factor, 1 + scale_factor, n_config)

        for i in range(n_config):
            frame = read(
                join_paths(self.main_directory, file_xyz_in.format(i)), format=format
            )
            cell = frame.get_cell()
            frame.set_cell(cell * scale[i], scale_atoms=True)
            write(
                join_paths(self.main_directory, file_xyz_out.format(i)),
                frame,
                format=format,
                columns=["symbols", "positions"],
            )

    def write_cp2k(
        self,
        structure_name: str,
        basis_set: str,
        potential: str,
        file_bash: str = "scripts/all.sh",
        file_batch: str = "scripts/{}.sh",
        file_input: str = "cp2k_input/{}.inp",
        file_xyz: str = "xyz/{}.xyz",
        n_config: int = None,
        nodes: int = 1,
        **kwargs
    ) -> str:
        """
        Writes .inp files and batch scripts for running cp2k from `n_config`
        .xyz files. Can set supported settings using `**kwargs`, in which case
        template file(s) will be formatted to contain the values provided. Note
        that the .xyz files should be in 'Ang'. Returns the command for running
        all cp2k files.

        Parameters
        ----------
        structure_name : str
            Name of the structure, used as part of the CP2K project name which in turn
            determines file names.
        basis_set : str
            Filepath to the CP2K basis set to use.
        potential : str
            Filepath to the CP2K potential to use.
        file_bash : str, optional
            File name to write a utility script which submits all of the batch
            scripts created by this function. Default is 'scripts/all.bash'.
        file_batch : str, optional
            Formatable file name to write the batch scripts to. Will be
            formatted with the frame number and any other `**kwargs`, so should
            contain '{}' as part of the string. There should already be a
            template of this file with 'template' instead of '{}' containing
            the details of the system that will remain constant across all
            frames and so do not need formatting. Default is 'scripts/{}.bat'.
        file_input : str, optional
            Formatable file name to write the cp2k input files to. Will be
            formatted with the frame number and any other `**kwargs`, so should
            contain '{}' as part of the string. There should already be a
            template of this file with 'template' instead of '{}' containing
            the details of the file that do not need formatting. Default is
            'cp2k_input/{}.inp'.
        file_xyz : str, optional
            Formatable file name to read the xyz files from. Will be formatted
            with the frame number, so should contain '{}' as part of the string.
            Default is 'xyz/{}.xyz'.
        n_config: int, optional
            The number of configuration frames to use. If the number provided
            is greater than the length of `self.trajectory`, or `n_config` is
            `None` then the length is used instead. Default is `None`.
        nodes: int, optional
            Number of nodes to request for the batch job. Default is `1`.
        **kwargs:
            Arguments to be used in formatting the cp2k input file. The
            template provided should contain a formatable string at the
            relevant location in the file. Each should be a tuple containing
            one or more values to run cp2k with:
            - cutoff: tuple of float
            - relcutoff: tuple of float

        Returns
        -------
        str
            Command to run `file_bash`, which in turn will submit all batch scripts.

        Examples
        --------
        To write 100 cp2k input files with specific values for `cutoff` and
        `relcutoff`:
            >>> from data import Data
            >>> d = Data('path/to/example_directory')
            >>> d.write_cp2k(n_config=100, cutoff=(600,), relcutoff=(60,))

        To create a "grid" of 20 input files with varying `cutoff` and
        `relcutoff` but for a single frame:
            >>> d.write_cp2k(n_config=1,
                            cutoff=(400, 500, 600, 700, 800),
                            relcutoff=(40, 50, 60, 70))
        """
        with open(
            join_paths(self.main_directory, file_input.format("template"))
        ) as f_template:
            input_template = f_template.read()

        with open(
            join_paths(self.main_directory, file_batch.format("template"))
        ) as f_template:
            batch_text_template = f_template.read()

        n_config = self._min_n_config(n_config)
        file_id_template = "n_{i}"

        if "cutoff" in kwargs:
            file_id_template += "_cutoff_{cutoff}"
            cutoff_values = kwargs["cutoff"]
            if isinstance(cutoff_values, float) or isinstance(cutoff_values, int):
                cutoff_values = [cutoff_values]
        else:
            cutoff_values = [None]

        if "relcutoff" in kwargs:
            file_id_template += "_relcutoff_{relcutoff}"
            relcutoff_values = kwargs["relcutoff"]
            if isinstance(relcutoff_values, float) or isinstance(relcutoff_values, int):
                relcutoff_values = [relcutoff_values]
        else:
            relcutoff_values = [None]

        batch_scripts = []
        for i in range(n_config):
            # Do not require job_array, so format with blank string
            format_dict = {"i": i, "job_array": ""}
            with open(join_paths(self.main_directory, file_xyz.format(i))) as f:
                header_line = f.readlines()[1]
                lattice_string = header_line.split('"')[1]
                lattice_list = lattice_string.split()
                format_dict["cell_x"] = " ".join(lattice_list[0:3])
                format_dict["cell_y"] = " ".join(lattice_list[3:6])
                format_dict["cell_z"] = " ".join(lattice_list[6:9])

            for cutoff in cutoff_values:
                for relcutoff in relcutoff_values:
                    format_dict["cutoff"] = cutoff
                    format_dict["relcutoff"] = relcutoff
                    format_dict["file_xyz"] = join_paths(
                        self.main_directory, file_xyz.format(i)
                    )
                    file_id = file_id_template.format(**format_dict)
                    file_input_formatted = join_paths(
                        self.main_directory, file_input.format(file_id)
                    )
                    with open(file_input_formatted, "w") as f:
                        f.write(
                            input_template.format(
                                file_id=file_id,
                                structure_name=structure_name,
                                basis_set=basis_set,
                                potential=potential,
                                **format_dict,
                            )
                        )

                    batch_scripts.append(
                        join_paths(self.main_directory, file_batch.format(file_id))
                    )

                    batch_text = batch_text_template.format(
                        nodes=nodes, job_name="CP2K", file_id=file_id, **format_dict
                    )
                    batch_text += "\nmpirun -np ${SLURM_NTASKS} cp2k.popt "
                    batch_text += "{0} &> ../cp2k_output/{1}_{2}.log" "".format(
                        file_input_formatted, structure_name, file_id
                    )
                    batch_text += (
                        "\nmv {0}_{1}-forces-1_0.xyz ../cp2k_output/{0}_{1}-forces-1_0.xyz"
                        "".format(structure_name, file_id)
                    )

                    with open(
                        join_paths(self.main_directory, file_batch.format(file_id)), "w"
                    ) as f:
                        f.write(batch_text)

        bash_text = "sbatch "
        bash_text += "\nsbatch ".join(batch_scripts)
        with open(join_paths(self.main_directory, file_bash), "w") as f:
            f.write(bash_text)

        return "bash {}".format(self.main_directory + file_bash)

    def write_qe_input(
        self,
        atoms: Atoms,
        frame_directory: str,
        structure: Structure,
        pseudos: List[str],
    ):
        """
        Writes the input file for Quantum Esspresso to `frame_directory` for the `atoms`
        provided.

        Parameters
        ----------
        atoms: Atoms
            The configuration to run QE on as an ASE Atoms object.
        frame_directory: str
            Complete path to the directory in which to write the QE input file.
        structure: Stucture
            The `Structure` being simulated. Will be used to determine the file name and
            get relevant elements.
        pseudos: list of str
            File names of the pseudo potentials to use, found within `pseudos` directory and
            ordered by atomic number.
        """
        cell = atoms.get_cell()
        symbols = structure.all_species.element_list
        masses = structure.all_species.mass_list
        with open(
            join_paths(frame_directory, "{}.in".format(structure.name)), "w"
        ) as f:
            print(
                """&control
  calculation = 'scf'
  tstress=.true
  tprnfor = .true.
  verbosity="high"
  pseudo_dir="../pseudos/"
  disk_io="low"
  outdir = './{}'
/

&system
  ibrav=0,
  nat={},
  ntyp=4
  ecutwfc=60
  ecutrho=400
  input_dft='pbe'
  occupations='fixed'
/

&electrons
  diagonalization='david'
  conv_thr=1.0e-8
  mixing_beta=0.7
/

&ions
  bfgs_ndim=3
/

CELL_PARAMETERS {{ angstrom }}
{:17.10f} {:17.10f} {:17.10f}
{:17.10f} {:17.10f} {:17.10f}
{:17.10f} {:17.10f} {:17.10f}

K_POINTS {{automatic}}
1 1 1 0 0 0

ATOMIC_SPECIES""".format(
                    structure.name,
                    len(atoms),
                    *cell[0],
                    *cell[1],
                    *cell[2],
                ),
                file=f,
            )

            for i in range(len(symbols)):
                print(
                    "{:2s} {:17.10f} {}".format(symbols[i], masses[i], pseudos[i]),
                    file=f,
                )

            print(
                """
ATOMIC_POSITIONS {{angstrom}}""",
                file=f,
            )

            for atom in atoms:
                print(
                    "{:2s} {:17.10f} {:17.10f} {:17.10f}".format(
                        atom.symbol, *atom.position
                    ),
                    file=f,
                )

    def write_qe_pp(self, frame_directory: str, structure: Structure):
        """
        Write "pp.in" input script for Quantum Espresso plottting.

        Parameters
        ----------
        frame_directory: str
            Complete path to the directory in which to write the QE input file.
        structure: Stucture
            The `Structure` being simulated. Will be used to determine the file name and
            get relevant elements.
        """
        with open(join_paths(frame_directory, "pp.in"), "w") as f:
            print(
                """&inputpp
    prefix  = 'pwscf'
    outdir = './{structure_name}'
    plot_num= 0
/
&plot
    nfile = 1
    iflag = 3
    output_format = 6
    fileout = '{structure_name}.cube'
/""".format(
                    structure_name=structure.name
                ),
                file=f,
            )

    def write_qe_slurm(self, frame_directory: str, structure: Structure):
        """
        Writes the batch script for running all stages of Quantum Espresso.

        Parameters
        ----------
        frame_directory: str
            Complete path to the directory in which to write the QE input file.
        structure: Stucture
            The `Structure` being simulated. Will be used to determine the file name and
            get relevant elements.
        """
        with open(join_paths(frame_directory, "qe.slurm"), "w") as f:
            print(
                """#!/usr/bin/env bash

#SBATCH -n 128
#SBATCH --exclusive
#SBATCH --reservation=scddevel
#SBATCH -t 2:00:00
#SBATCH -C amd

module purge

module use /work3/cse/dlp//scd/modules/all
module load QuantumESPRESSO/6.6-foss-2020a
n=$SLURM_NTASKS

export OMP_NUM_THREADS=1

cd {frame_directory}
mpirun -n $n pw.x  -i  {structure_name}.in > {structure_name}.log
mpirun -n 32 pp.x  <  pp.in > {structure_name}-pp.log
/home/vol02/scarf562/bin/bader {structure_name}.cube > bader.log
rm -f {structure_name}.cube
rm -rf ./{structure_name}/pwscf.save
rm -f tmp.pp
""".format(
                    frame_directory=frame_directory, structure_name=structure.name
                ),
                file=f,
            )

        return "sbatch {}\n".format(join_paths(frame_directory, "qe.slurm"))

    def prepare_qe(
        self,
        temperatures: List[int],
        pressures: List[int],
        structure: Structure,
        pseudos: List[str],
        qe_directory: str = "qe",
        selection: Tuple[int, int] = (0, 1),
    ):
        """
        Prepare input and batch scripts needed for Quantum Espresso. Expects a trajectory file
        to be in `qe_directory` with the naming patten
        "{structure.name}-T{temperature}-p{pressure}.xyz".

        Parameters
        ----------
        qe_directory: str
            Directory for all Quantum Espresso files and sub directories.
        temperatures: list of int
            All temperatures to run Quantum Espresso for.
        pressures: list of int
            All pressures to run Quantum Espresso for.
        structure: Structure
            The Structure which is being simulated.
        pseudos: list of str
            File names of the pseudo potentials to use, found within `pseudos` directory and
            ordered by atomic number.
        selection: tuple of int
            Allows for subsampling of the trajectory files. First entry is the index of the
            first frame to use, second allows for every nth frame to be selected.

        """
        submit_all_text = ""
        for t in temperatures:
            for p in pressures:
                traj = read(
                    join_paths(
                        self.main_directory,
                        qe_directory,
                        structure.name + "-T" + str(t) + "-p" + str(p) + ".xyz",
                    ),
                    index=":",
                )
                for j, a in enumerate(traj):
                    if j >= selection[0] and j % selection[1] == 0:
                        folder = join_paths(
                            self.main_directory,
                            qe_directory,
                            "T{:d}-p{}-{:d}".format(t, p, j),
                        )
                        if not exists(folder):
                            mkdir(folder)
                        self.write_qe_input(
                            a, folder, structure=structure, pseudos=pseudos
                        )
                        self.write_qe_pp(folder, structure=structure)
                        submit_all_text += self.write_qe_slurm(folder, structure)

        if submit_all_text:
            with open(join_paths(self.scripts_directory, "qe_all.sh"), "w") as f:
                f.write(submit_all_text)

    def print_cp2k_table(
        self,
        file_output: str = "cp2k_output/{}.log",
        n_config: int = None,
        n_mpi: int = 1,
        **kwargs
    ):
        """
        Print the final energy, time taken, and grid allocation for given cp2k
        settings. Formatted for a .md table.

        Parameters
        ----------
        file_output : str, optional
            Formatable file name to read the cp2k output from. Will be
            formatted with the frame number and any other `**kwargs`, so should
            contain '{}' as part of the string.
            Default is 'cp2k_output/{}.log'.
        n_config: int, optional
            The number of configuration frames to use. If the number provided
            is greater than the length of `self.trajectory`, or `n_config` is
            `None` then the length is used instead. Default is `None`.
        n_mpi: int, optional
            The number of MPI processes used for cp2k. Default is `1`.
        **kwargs:
            Arguments to be used in formatting the name of the cp2k output file(s).
            Each should be a tuple containing one or more values:
              - cutoff: tuple of float
              - relcutoff: tuple of float

        Examples
        --------
        >>> from data import Data
        >>> d = Data('path/to/example_directory')
        >>> d.print_cp2k_table(file_output='cutoff/cp2k_output/{}.log',
                               n_config=1,
                               cutoff=(400, 500, 600, 700, 800, 900, 1000,
                                       1100, 1200, 1300, 1400, 1500, 1600,
                                       1700, 1800, 1900, 2000, 2100, 2200,
                                       2300, 2400))
        """
        n_config = self._min_n_config(n_config)
        file_id_template = "n_{i}"

        header = "| "
        msg = "| "
        if "cutoff" in kwargs:
            file_id_template += "_cutoff_{cutoff}"
            cutoff_values = kwargs["cutoff"]
            header += "Cutoff | "
            msg += "  {cutoff} | "
        else:
            cutoff_values = [None]

        if "relcutoff" in kwargs:
            file_id_template += "_relcutoff_{relcutoff}"
            relcutoff_values = kwargs["relcutoff"]
            header += " Relative Cutoff | "
            msg += "  {relcutoff} | "
        else:
            relcutoff_values = [None]

        # Print table header:
        print(
            header + "Processes | Energy                | t/step (s) | t total (s) "
            "| Grid 1 | Grid 2 | Grid 3 | Grid 4 |"
        )

        for i in range(n_config):
            for cutoff in cutoff_values:
                for relcutoff in relcutoff_values:
                    format_dict = {"i": i, "cutoff": cutoff, "relcutoff": relcutoff}

                    file_id = file_id_template.format(**format_dict)
                    with open(
                        join_paths(self.main_directory, file_output.format(file_id))
                    ) as f:
                        energy = None
                        m_grid = []
                        steps = None
                        total_time = None
                        for line in f.readlines():
                            if re.search("^ ENERGY", line):
                                energy = line.split()[-1]
                            elif re.search("^ count for grid", line):
                                m_grid.append(line.split()[4])
                            elif re.search("^  outer SCF loop converged in", line):
                                steps = line.split()[-2]
                            elif re.search("^ CP2K  ", line):
                                total_time = line.split()[-2]

                        try:
                            time_per_step = round(float(total_time) / int(steps), 1)
                        except TypeError:
                            time_per_step = None
                        msg_out = msg.format(**format_dict)
                        msg_out += (
                            "        {n_mpi} | {energy} | {time_per_step} |    "
                            "{total_time} | ".format(
                                n_mpi=n_mpi,
                                energy=energy,
                                time_per_step=time_per_step,
                                total_time=total_time,
                                **format_dict,
                            )
                        )
                        msg_out += " | ".join(m_grid)
                        print(msg_out + " |")

    def read_charges_qe(self, acf_file: str, n_atoms: int):
        """
        Reads the atomic charges from the ACF file output from Quantum Espresso.

        Parameters
        ----------
        acf_file: str
            Complete file path to read from in the .acf format.
        n_atoms:
            The number of atoms present in the frame in question.
        """
        charges = []
        with open(acf_file) as f:
            f.readline()
            f.readline()
            k = 0
            for line in f:
                k = k + 1
                aid, _, _, _, ch, _, _ = line.split()
                charges += [float(ch)]
                if k == n_atoms:
                    break
            return charges

    def write_n2p2_data_qe(
        self,
        structure_name: str,
        temperatures: List[int],
        pressures: List[int],
        valences: Dict[str, int],
        qe_directory: str = "qe",
        file_qe_log: str = "T{temperature}-p{pressure}-{index}/{structure_name}.log",
        file_qe_charges: str = "T{temperature}-p{pressure}-{index}/ACF.dat",
        file_xyz: str = "{structure_name}-T{temperature}-p{pressure}.xyz",
        file_n2p2_input: str = "input.data",
        n2p2_units: dict = None,
    ):
        """
        Read the Quantum Espresso output files, then format the information for n2p2 and write
        to file.

        Parameters
        ----------
        structure_name: str
            The name of the Structure in question.
        temperatures: list of int
            All temperatures that Quantum Espresso was run for.
        pressures: list of int
            All pressures that Quantum Espresso was run for.
        valences: dict of str, int
            The valences of the species comprising `strucuture`. The keys should be the
            chemical symbols, with positive int values.
        qe_directory: str
            Directory for all Quantum Espresso files and sub directories. Default is "qe".
        file_qe_log: str
            File path for the Quantum Espresso log file, relative to `qe_directory`. Should be
            formatable with the `temperature`, `pressure`, `index` and `structure_name`.
            Default is "T{temperature}-p{pressure}-{index}/{structure_name}.log".
        file_qe_charges: str
            File path for the Quantum Espresso charge file, relative to `qe_directory`.
            Should be formatable with the `temperature`, `pressure` and `index`.
            Default is "T{temperature}-p{pressure}-{index}/ACF.dat".
        file_xyz: str
            File path for the initial trajectory file, relative to `qe_directory`. Should be
            formatable with the `temperature`, `pressure` and `structure_name`.
            Default is "{structure_name}-T{temperature}-p{pressure}.xyz".
        file_n2p2_input: str
            File path to write the n2p2 data to, relative to `self.n2p2_directory`.
            Default is "input.data".
        n2p2_units: dict, optional
            The units to use for n2p2. No specific units are required, however
            they should be consistent (i.e. positional data and symmetry
            functions can use 'Ang' or 'Bohr' provided both do). Default is `None`, which will
            lead to `{'length': 'Bohr', 'energy': 'Ha', 'force': 'Ha / Bohr'}` being used.
        """
        if structure_name not in self.structure_names:
            raise ValueError(
                "`structure_name` {} not recognized".format(structure_name)
            )

        text = ""
        if n2p2_units is None:
            n2p2_units = {"length": "Bohr", "energy": "Ha", "force": "Ha / Bohr"}

        for temperature in temperatures:
            for pressure in pressures:
                trajectory = read(
                    join_paths(
                        self.main_directory,
                        qe_directory,
                        file_xyz.format(
                            structure_name=structure_name,
                            temperature=temperature,
                            pressure=pressure,
                        ),
                    ),
                    index=":",
                )
                for index, frame in enumerate(trajectory):
                    full_filepath = join_paths(
                        self.main_directory,
                        qe_directory,
                        file_qe_log.format(
                            structure_name=structure_name,
                            temperature=temperature,
                            pressure=pressure,
                            index=index,
                        ),
                    )
                    charge_filepath = join_paths(
                        self.main_directory,
                        qe_directory,
                        file_qe_charges.format(
                            temperature=temperature,
                            pressure=pressure,
                            index=index,
                        ),
                    )
                    if exists(full_filepath):
                        if n2p2_units["length"] != "Ang":
                            frame.set_cell(
                                frame.get_cell() / self.units[n2p2_units["length"]],
                                scale_atoms=True,
                            )

                        if exists(charge_filepath):
                            charges = self.read_charges_qe(charge_filepath, len(frame))
                        else:
                            charges = None

                        with open(full_filepath) as f:
                            lines = f.readlines()
                        for i, line in enumerate(lines):
                            if line.startswith("!    total energy"):
                                energy = (
                                    float(line.split()[4])
                                    * self.units[line.split()[5]]
                                    / self.units[n2p2_units["energy"]]
                                )
                            elif line.startswith("     Forces acting on atoms"):
                                forces = [
                                    force_line.split()[-3:]
                                    for force_line in lines[i + 2 : i + 2 + len(frame)]
                                ]
                                forces = np.array(forces).astype(float)
                                qe_force_unit = line.split()[-1][:-2]
                                forces *= self.units[qe_force_unit.split("/")[0]]
                                forces /= self.units[qe_force_unit.split("/")[1]]
                                forces /= self.units[n2p2_units["energy"]]
                                forces *= self.units[n2p2_units["length"]]

                        text += "begin\n"
                        text += "comment frame_index={0} units={1}\n".format(
                            index, n2p2_units
                        )
                        text += "comment structure {}\n".format(structure_name)
                        for vector in frame.get_cell():
                            text += "lattice {} {} {}\n".format(*vector)

                        symbols = frame.get_chemical_symbols()
                        positions = frame.get_positions()
                        for i in range(len(frame)):
                            if charges is None or len(frame) > len(charges):
                                charge = 0.0
                            else:
                                charge = valences[symbols[i]] - charges[i]
                            text += "atom {} {} {} {} {} 0.0 {} {} {}\n".format(
                                *positions[i],
                                symbols[i],
                                charge,
                                *forces[i],
                            )

                        text += "energy {}\n".format(energy)
                        text += "charge {}\n".format(0.0)
                        text += "end\n"
                    else:
                        print("{} not found, skipping".format(full_filepath))

        if len(text) == 0:
            raise IOError("No files found.")

        if isfile(join_paths(self.n2p2_directory, file_n2p2_input)):
            with open(join_paths(self.n2p2_directory, file_n2p2_input), "a") as f:
                f.write(text)
        else:
            with open(join_paths(self.n2p2_directory, file_n2p2_input), "w") as f:
                f.write(text)

    def write_n2p2_data(
        self,
        structure_name: str,
        file_cp2k_out: str = "cp2k_output/{}.log",
        file_cp2k_forces: str = "cp2k_output/{}-forces-1_0.xyz",
        file_xyz: str = "xyz/{}.xyz",
        file_n2p2_input: str = "input.data",
        n_config: int = None,
        n2p2_units: dict = None,
    ):
        """
        Reads xyz and cp2k output data, and writes it to file as n2p2 input
        data in the following format:

            begin
            comment <comment>
            lattice <ax> <ay> <az>
            lattice <bx> <by> <bz>
            lattice <cx> <cy> <cz>
            atom <x1> <y1> <z1> <e1> <c1> <n1> <fx1> <fy1> <fz1>
            atom <x2> <y2> <z2> <e2> <c2> <n2> <fx2> <fy2> <fz2>
            ...
            atom <xn> <yn> <zn> <en> <cn> <nn> <fxn> <fyn> <fzn>
            energy <energy>
            charge <charge>
            end

        If `file_input` already exists, then it will not be overwritten but
        appended to. This allows multiple directories of xyz and cp2k output to
        be combined into one n2p2 input file.

        Parameters
        ----------
        structure_name : str
            Name of the structure, used when running active learning.
        file_cp2k_out : str, optional
            Formatable file name to read the cp2k output from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
            Default is 'cp2k_output/{}.log'.
        file_cp2k_forces : str, optional
            Formatable file name to read the cp2k forces from. Will be formatted with the
            frame number, so should contain '{}' as part of the string.
            Default is 'cp2k_output/{}-forces-1_0.xyz'.
        file_xyz : str, optional
            Formatable file name to read the xyz files from. Will be formatted with the
            frame number, so should contain '{}' as part of the string. Default is 'xyz/{}.xyz'.
        file_n2p2_input : str, optional
            File name to write the n2p2 data to relative to `self.n2p2_directory`.
            Default is 'input.data'.
        n_config: int, optional
            The number of configuration frames to use. If the number provided
            is greater than the length of `self.trajectory`, or `n_config` is
            `None` then the length is used instead. Default is `None`.
        n2p2_units: dict, optional
            The units to use for n2p2. No specific units are required, however
            they should be consistent (i.e. positional data and symmetry
            functions can use 'Ang' or 'Bohr' provided both do). Default is `None`, which will
            lead to `{'length': 'Bohr', 'energy': 'Ha', 'force': 'Ha / Bohr'}` being used.

        Examples
        --------
        To write the output from two directories to single n2p2 input:
            >>> from data import Data
            >>> d = Data('/path/to/examples/')
            >>> d.write_n2p2_data(file_cp2k_out='example_1/cp2k_output/{}.log',
            >>>                   file_cp2k_forces='example_1/cp2k_output/{}-forces-1_0.xyz',
            >>>                   file_xyz='example_1/xyz/{}.xyz',
            >>>                   file_input='examples_combined/n2p2/input.data',
            >>>                   n_config=101)
            >>> d.write_n2p2_data(file_cp2k_out='example_2/cp2k_output/{}.log',
            >>>                   file_cp2k_forces='example_2/cp2k_output/{}-forces-1_0.xyz',
            >>>                   file_xyz='example_2/xyz/{}.xyz',
            >>>                   file_input='examples_combined/n2p2/input.data',
            >>>                   n_config=101)
        """
        if structure_name not in self.structure_names:
            raise ValueError(
                "`structure_name` {} not recognized".format(structure_name)
            )

        text = ""
        if n2p2_units is None:
            n2p2_units = {"length": "Bohr", "energy": "Ha", "force": "Ha / Bohr"}
        n = self._min_n_config(n_config)
        for i in range(n):
            with open(join_paths(self.main_directory, file_xyz.format(i))) as f:
                xyz_lines = f.readlines()
                n_atoms = int(xyz_lines[0].strip())
                header_list = xyz_lines[1].split('"')
                lattice_list = header_list[1].split()
                if n2p2_units["length"] != "Ang":
                    for j, lattice in enumerate(lattice_list):
                        lattice_list[j] = (
                            float(lattice) / self.units[n2p2_units["length"]]
                        )

            with open(join_paths(self.main_directory, file_cp2k_forces.format(i))) as f:
                force_lines = f.readlines()

            with open(join_paths(self.main_directory, file_cp2k_out.format(i))) as f:
                energy = None
                lines = f.readlines()
                for j, line in enumerate(lines):
                    if re.search("^ ENERGY", line):
                        energy = line.split()[-1]
                    if re.search("Hirshfeld Charges", line):
                        charge_lines = lines[j + 3 : j + 3 + n_atoms]
                        total_charge = lines[j + 3 + n_atoms + 1].split()[-1]

            if energy is None:
                raise ValueError("Energy not found in {}".format(file_cp2k_out))
            if n2p2_units["energy"] != "Ha":
                # cp2k output is in Ha
                energy = (
                    float(energy) * self.units["Ha"] / self.units[n2p2_units["energy"]]
                )

            text += "begin\n"
            text += "comment config_index={0} units={1}\n".format(i, n2p2_units)
            text += "comment structure {}\n".format(structure_name)
            text += "lattice {0} {1} {2}\n".format(
                lattice_list[0], lattice_list[1], lattice_list[2]
            )
            text += "lattice {0} {1} {2}\n".format(
                lattice_list[3], lattice_list[4], lattice_list[5]
            )
            text += "lattice {0} {1} {2}\n".format(
                lattice_list[6], lattice_list[7], lattice_list[8]
            )
            for j in range(n_atoms):
                atom_xyz = xyz_lines[j + 2].split()
                for k, position in enumerate(atom_xyz[1:], 1):
                    atom_xyz[k] = float(position) / self.units[n2p2_units["length"]]

                force = force_lines[j + 4].split()[-3:]
                if n2p2_units["force"] != "Ha / Bohr":
                    # cp2k output is in Ha / Bohr
                    for i in range(len(force)):
                        force[i] = (
                            float(force[i])
                            * self.units["Ha"]
                            / self.units["Bohr"]
                            / self.units[n2p2_units["energy"]]
                        )

                charge = charge_lines[j].split()[-1]
                text += "atom {1} {2} {3} {0} {4} 0.0 {5} {6} {7}\n".format(
                    *atom_xyz + [charge] + force
                )

            text += "energy {}\n".format(energy)
            text += "charge {}\n".format(total_charge)
            text += "end\n"

        if isfile(join_paths(self.n2p2_directory, file_n2p2_input)):
            with open(join_paths(self.n2p2_directory, file_n2p2_input), "a") as f:
                f.write(text)
        else:
            with open(join_paths(self.n2p2_directory, file_n2p2_input), "w") as f:
                f.write(text)

    def write_n2p2_nn(
        self,
        r_cutoff: float,
        type: str,
        rule: str,
        mode: str,
        n_pairs: int,
        zetas: list = None,
        r_lower: float = None,
        r_upper: float = None,
        file_nn_template: str = "input.nn.template",
        file_nn: str = "input.nn",
    ):
        """
        Based on `file_template`, write the input.nn file for n2p2 with
        symmetry functions generated using the provided arguments. File locations should be
        relative to `self.n2p2_directory`.

        Note that all distances (`r_cutoff`, `r_lower`, `r_upper`) should have
        the same units as the n2p2 `input.data` file (by default, Bohr).

        Parameters
        ----------
        r_cutoff: float
            The cutoff distance for the symmetry functions.
        type: {'radial', 'angular_narrow', 'angular_wide',
               'weighted_radial', 'weighted_angular'}
            The type of symmetry function to generate.
        rule: {'imbalzano2018', 'gastegger2018'}
            The ruleset used to determine how to chose values for r_shift and eta.
        mode: {'center', 'shift'}
            Whether the symmetry functions are centred or are shifted relative
            to the central atom.
        n_pairs: int
            The number of symmetry functions to generate. Specifically,
            `n_pairs` values for eta and r_shift are generated.
        zetas: list of int, optional
            Not used for radial functions. Default is `[]`.
        r_lower: float, optional
            Not used for the 'imbalzano2018' ruleset. For 'gastegger2018', this
            sets either the minimum r_shift value or the maximum eta value for
            modes 'shift' and 'center' respectively. Default is `None`.
        r_upper: float, optional
            Not used for the 'imbalzano2018' ruleset. For 'gastegger2018', this
            sets either the maximum r_shift value or the minimum eta value for
            modes 'shift' and 'center' respectively. Default is `None`.
        file_nn_template : str, optional
            The file to read the general network architecture from (i.e.
            everything except the symmetry functions). Default is
            'n2p2/input.nn.template'.
        file_nn: str, optional
            The file to write the output to. If the file already exists, then
            it is appended to with the new symmetry functions. If it does not,
            then it is created and the text from `file_nn_template` is written to
            it before the symmetry functions are written. Default is
            'n2p2/input.nn'..

        Examples
        --------
        To write multiple sets of symmetry functions to the same file:
            >>> from data import Data
            >>> d = Data('path/to/example_directory')
            >>> d.write_n2p2_nn(r_cutoff=12.0,
                                type='radial',
                                rule='imbalzano2018',
                                mode='center',
                                n_pairs=10)
            >>> d.write_n2p2_nn(r_cutoff=12.0,
                                type='angular_narrow',
                                rule='imbalzano2018',
                                mode='center',
                                n_pairs=10,
                                zetas=[1, 4])
            >>> d.write_n2p2_nn(r_cutoff=12.0,
                                type='angular_wide',
                                rule='imbalzano2018',
                                mode='center',
                                n_pairs=10,
                                zetas=[1, 4])
        """
        if zetas is None:
            zetas = []

        generator = SymFuncParamGenerator(elements=self.elements, r_cutoff=r_cutoff)
        generator.symfunc_type = type
        generator.zetas = zetas

        generator.generate_radial_params(
            rule=rule,
            mode=mode,
            nb_param_pairs=n_pairs,
            r_lower=r_lower,
            r_upper=r_upper,
        )

        if isfile(join_paths(self.n2p2_directory, file_nn)):
            with open(join_paths(self.n2p2_directory, file_nn), "a") as f:
                generator.write_settings_overview(fileobj=f)
                generator.write_parameter_strings(fileobj=f)
        else:
            with open(join_paths(self.n2p2_directory, file_nn_template)) as f:
                template_text = f.read()
            with open(join_paths(self.n2p2_directory, file_nn), "w") as f:
                f.write(template_text)
                generator.write_settings_overview(fileobj=f)
                generator.write_parameter_strings(fileobj=f)

    def write_n2p2_scripts(
        self,
        normalise: bool = False,
        n_scaling_bins: int = 500,
        range_threshold: float = 1e-4,
        nodes: int = 1,
        file_batch_template: str = "scripts/template.sh",
        file_prune: str = "scripts/n2p2_prune.sh",
        file_train: str = "scripts/n2p2_train.sh",
        file_nn: str = "input.nn",
    ):
        """
        Write batch script for scaling and pruning symmetry functions, and for
        training the network. Returns the command to submit the scripts.

        Parameters
        ----------
        normalise: bool, optional
            Whether to apply normalisation to the network. Default is `False`.
        n_scaling_bins: int, optional
            Number of bins for symmetry function histograms. Default is `500`.
        range_threshold: float, optional
            Symmetry functions with ranges below this will be "pruned" and not
            used in the training. Default is `1e-4`.
        nodes: int, optional
            Number of nodes to request for the batch job. Default is `1`.
        file_batch_template: str, optional
            File location of template to use for batch scripts. Default is
            'scripts/template.sh'.
        file_prune: str, optional
            File location to write scaling and pruning batch script to.
            Default is 'scripts/n2p2_prune.sh'.
        file_train: str, optional
            File location to write training batch script to.
            Default is 'scripts/n2p2_train.sh'.
        file_nn: str, optional
            File location of n2p2 file defining the neural network. Default is
            "input.nn".
        """
        with open(join_paths(self.main_directory, file_batch_template)) as f:
            batch_template_text = f.read()

        format_dict = {"job_name": "n2p2_scale_prune", "nodes": nodes, "job_array": ""}
        output_text = batch_template_text.format(**format_dict)
        output_text += "\ncd {}".format(self.n2p2_directory)
        if normalise:
            output_text += "\nmpirun -np ${SLURM_NTASKS} " + join_paths(
                self.n2p2_bin, "nnp-norm"
            )
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + join_paths(self.n2p2_bin, "nnp-scaling")
            + " "
            + str(n_scaling_bins)
        )
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + join_paths(self.n2p2_bin, "nnp-prune")
            + " range "
            + str(range_threshold)
        )
        output_text += "\nmv {0} {0}.unpruned".format(file_nn)
        output_text += "\nmv output-prune-range.nn {}".format(file_nn)
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + join_paths(self.n2p2_bin, "nnp-scaling")
            + " "
            + str(n_scaling_bins)
        )

        with open(join_paths(self.main_directory, file_prune), "w") as f:
            f.write(output_text)

        format_dict["job_name"] = "n2p2_train"
        output_text = batch_template_text.format(**format_dict)
        output_text += "\ncd {}".format(self.n2p2_directory)
        output_text += "\nmpirun -np ${SLURM_NTASKS} " + join_paths(
            self.n2p2_bin, "nnp-train"
        )

        with open(join_paths(self.main_directory, file_train), "w") as f:
            f.write(output_text)

    def write_lammps_data(
        self,
        file_xyz: str,
        file_data: str = "lammps.data",
        lammps_unit_style: str = "electron",
    ):
        """
        Reads data from xyz format and writes it for LAMMPS.

        Parameters
        ----------
        file_xyz : str
            Complete file name to read the xyz positions from.
        file_data: str, optional
            File name to write the LAMMPS formatted positions to.
            Default is 'lammps.data'.
        lammps_unit_style: str, optional
            The LAMMPS unit system to use. The xyz is assumed to be in ASE
            default units (i.e. 'Ang' for length). Default is 'electron'.
        """
        format_in = "extxyz"
        format_out = "lammps-data"
        atoms = read(join_paths(self.main_directory, file_xyz), format=format_in)
        write(
            join_paths(self.lammps_directory, file_data),
            atoms,
            format=format_out,
            units=lammps_unit_style,
        )

    def write_lammps_pair(
        self,
        r_cutoff: float,
        file_lammps_template: str = "lammps/template.lmp",
        file_out: str = "lammps/md.lmp",
        n2p2_directory: str = "n2p2",
        lammps_unit_style: str = "electron",
    ):
        """
        Writes the pair commands 'pair_style' and 'pair_coeff' to `file_out`,
        with the rest of the LAMMPS commands to be contained in `file_lammps_template`
        along with '{pair_commands}' to allow formatting that section to the
        following:

            pair_style nnp dir {n2p2_directory} showew no showewsum 10 resetew no maxew 100 \
                emap {elements_map}
            pair_coeff * * {r_cutoff}

        `r_cutoff` must be given in the LAMMPS units regardless of what was used for training.

        Parameters
        ----------
        r_cutoff: float
            The cutoff distance for the symmetry functions.
        file_lammps_template: str, optional
            Complete file name of the template for `file_out`. Should contain
            '{pair_commands}' to allow formatting. Default is
            'lammps/template.lmp'.
        file_out: str, optional
            Complete file name of the output file. Default is 'lammps/md.lmp'.
        n2p2_directory: str, optional
            Directory containing all the n2p2 files needed by the LAMMPS/n2p2
            interface. Default is 'n2p2'.
        lammps_unit_style: str, optional
            The LAMMPS unit system to use. Default is 'electron'.
        """
        with open(join_paths(self.main_directory, file_lammps_template)) as f:
            template_text = f.read()

        elements = self.elements
        elements.sort()
        elements_map = '"'
        for i, element in enumerate(elements):
            elements_map += "{0}:{1},".format(i + 1, element)
        elements_map = elements_map[:-1] + '"'

        pair_style = (
            "pair_style nnp dir {0} showew no showewsum 10 resetew no maxew 100 emap {1}"
            "".format(n2p2_directory, elements_map)
        )

        lammps_units = {
            "electron": {"length": 1, "energy": 1},
            "real": {
                "length": 1 / self.units["Bohr"],
                "energy": self.units["kcal"] / (self.units["Ha"] * self.units["mol"]),
            },
            "metal": {"length": 1 / self.units["Bohr"], "energy": 1 / self.units["Ha"]},
            "si": {
                "length": 1e10 / self.units["Bohr"],
                "energy": self.units["kJ"] / (self.units["Ha"] * 1e3),
            },
            "cgs": {
                "length": 1e8 / self.units["Bohr"],
                "energy": self.units["kJ"] / (self.units["Ha"] * 1e10),
            },
            "micro": {
                "length": 1e4 / self.units["Bohr"],
                "energy": self.units["kJ"] / (self.units["Ha"] * 1e15),
            },
            "nano": {
                "length": 1e1 / self.units["Bohr"],
                "energy": self.units["kJ"] / (self.units["Ha"] * 1e21),
            },
        }

        if lammps_unit_style == "electron":
            # No conversion required
            pass
        elif lammps_unit_style in lammps_units:
            pair_style += " cflength {length} cfenergy {energy}".format(
                **lammps_units[lammps_unit_style]
            )
        else:
            raise ValueError(
                "`lammps_unit_style={}` not recognised".format(lammps_unit_style)
            )

        pair_coeff = "pair_coeff * * {}".format(r_cutoff)
        output_text = template_text.format(
            pair_commands=pair_style + "\n" + pair_coeff,
            lammps_unit_style=lammps_unit_style,
        )

        with open(join_paths(self.main_directory, file_out), "w") as f:
            f.write(output_text)

    def _write_active_learning_lammps_script(
        self,
        n_simulations: int,
        nodes: int = 1,
        file_batch_template: str = "template.sh",
        file_batch_out: str = "active_learning_lammps.sh",
    ):
        """
        Write batch script for using LAMMPS to generate configurations for active learning.

        Parameters
        ----------
        nodes: int
            Number of simulations required. This is used to set the upper limit on the SLURM
            job array.
        nodes: int, optional
            Number of nodes to request for the batch job. Default is `1`.
        file_batch_template: str, optional
            File location of template to use for batch scripts relative to
            `scripts_sub_directory`. Default is 'scripts/template.sh'.
        file_batch_out: str, optional
            File location to write the batch script relative to `scripts_sub_directory`.
            Default is 'scripts/active_learning_lammps.sh'.
        """
        with open(join_paths(self.scripts_directory, file_batch_template)) as f:
            batch_template_text = f.read()

        format_dict = {
            "job_name": "active_learning_LAMMPS",
            "nodes": nodes,
            "job_array": "#SBATCH --array=1-{}".format(n_simulations),
        }
        output_text = batch_template_text.format(**format_dict)
        output_text += "\nulimit -s unlimited"
        output_text += (
            "\npath=$(sed -n ${SLURM_ARRAY_TASK_ID}p ${SLURM_SUBMIT_DIR}/"
            + self.active_learning_directory
            + "/joblist_mode1.dat)"
        )
        output_text += "\ndir=$(date '+%Y%m%d_%H%M%S_%N')"
        output_text += "\nmkdir -p /scratch/$(whoami)/${dir}"
        output_text += (
            "\nrsync -a ${SLURM_SUBMIT_DIR}/"
            + self.active_learning_directory
            + "/mode1/${path}/* /scratch/$(whoami)/${dir}/"
        )
        output_text += "\ncd /scratch/$(whoami)/${dir}"
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + self.lammps_executable
            + " -in input.lammps -screen none"
        )
        output_text += "\nrm -r /scratch/$(whoami)/${dir}/RuNNer"
        output_text += (
            "\nrsync -a /scratch/$(whoami)/${dir}/* ${SLURM_SUBMIT_DIR}/"
            + self.active_learning_directory
            + "/mode1/${path}/"
        )
        output_text += "\nrm -r /scratch/$(whoami)/${dir}"

        with open(join_paths(self.scripts_directory, file_batch_out), "w") as f:
            f.write(output_text)
            print(
                "Batch script written to {}".format(
                    join_paths(self.scripts_directory, file_batch_out)
                )
            )

    def _write_active_learning_nn_script(
        self,
        n2p2_directories: List[str],
        nodes: int = 1,
        file_batch_template: str = "template.sh",
        file_batch_out: str = "active_learning_nn.sh",
    ):
        """
        Write batch script for using the neural network to calculate energies for
        configurations as part of the active learning.

        Parameters
        ----------
        n2p2_directories : list of str
            List of directories to compare. Should have exactly 2 entries.
        nodes: int, optional
            Number of nodes to request for the batch job. Default is `1`.
        file_batch_template: str, optional
            File location of template to use for batch scripts relative to
            `scripts_sub_directory`. Default is 'template.sh'.
        file_batch_out: str, optional
            File location to write the batch script to relative to
            `scripts_sub_directory`. Default is 'active_learning_nn.sh'.
        """
        with open(join_paths(self.scripts_directory, file_batch_template)) as f:
            batch_template_text = f.read()

        # Format SBATCH variables
        format_dict = {
            "job_name": "active_learning_NN",
            "nodes": nodes,
            "job_array": "#SBATCH --array=1-2",
        }
        output_text = batch_template_text.format(**format_dict)

        # Setup
        output_text += "\nn2p2_directories=({} {})".format(*n2p2_directories)
        output_text += "\nmkdir {}/mode2".format(self.active_learning_directory)
        output_text += (
            "\nmkdir {}/mode2".format(self.active_learning_directory)
            + "/HDNNP_${SLURM_ARRAY_TASK_ID}"
        )
        output_text += (
            "\ncp ${n2p2_directories[${SLURM_ARRAY_TASK_ID} - 1]}/input.nn "
            + "{}/mode2".format(self.active_learning_directory)
            + "/HDNNP_${SLURM_ARRAY_TASK_ID}"
        )
        output_text += (
            "\ncp ${n2p2_directories[${SLURM_ARRAY_TASK_ID} - 1]}/scaling.data "
            + "{}/mode2".format(self.active_learning_directory)
            + "/HDNNP_${SLURM_ARRAY_TASK_ID}"
        )
        output_text += (
            "\ncp ${n2p2_directories[${SLURM_ARRAY_TASK_ID} - 1]}/weights.*.data "
            + "{}/mode2".format(self.active_learning_directory)
            + "/HDNNP_${SLURM_ARRAY_TASK_ID}"
        )
        output_text += (
            "\nsed -i s/'.*test_fraction.*'/'test_fraction 0.0'/g "
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )
        output_text += (
            "\nsed -i s/'epochs.*'/'epochs 0'/g "
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )
        output_text += (
            "\nsed -i s/'.*use_old_weights_short'/'use_old_weights_short'/g "
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )
        output_text += (
            "\nsed -i s/'.*use_short_forces'/'use_short_forces'/g"
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )
        output_text += (
            "\nsed -i s/'.*write_trainpoints'/'write_trainpoints'/g "
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )
        output_text += (
            "\nsed -i s/'.*write_trainforces'/'write_trainforces'/g "
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )
        output_text += (
            "\nsed -i s/'.*precondition_weights'/'#precondition_weights'/g "
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )
        output_text += (
            "\nsed -i s/'.*nguyen_widrow_weights_short'/'#nguyen_widrow_weights_short'/g "
            "{}/mode2/HDNNP_*/input.nn".format(self.active_learning_directory)
        )

        # Train
        output_text += (
            "\ncd {}/mode2".format(self.active_learning_directory)
            + "/HDNNP_${SLURM_ARRAY_TASK_ID}"
        )
        output_text += "\nln -s ../../input.data-new input.data"
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + "{}/nnp-train > mode_2.out".format(self.n2p2_bin)
        )

        with open(join_paths(self.scripts_directory, file_batch_out), "w") as f:
            f.write(output_text)
            print(
                "Batch script written to {}".format(
                    join_paths(self.scripts_directory, file_batch_out)
                )
            )

    def choose_weights(
        self,
        epoch: int = None,
        minimum_criterion: str = None,
        file_out: str = "weights.{0:03d}.data",
    ):
        """
        Copies the weights files for a particular epoch from "weights.XXX.YYYYYY.out" to
        `file_out`.

        By default, the most recent epoch is chosen. Optionally, a specific epoch can be
        chosen with `epoch` or one will be automatically based on the `minimum_criterion`.

        Parameters
        ----------
        epoch : int, optional
            The epoch to copy weights for. Default is `None`.
        minimum_criterion : str, optional
            Defines what value to select the epoch based on. Possible values are:
              - "RMSEpa_Etrain_pu" RMSE of training energies per atom (physical units)
              - "RMSEpa_Etest_pu"  RMSE of test energies per atom (physical units)
              - "RMSE_Etrain_pu"   RMSE of training energies (physical units)
              - "RMSE_Etest_pu"    RMSE of test energies (physical units)
              - "MAEpa_Etrain_pu"  MAE of training energies per atom (physical units)
              - "MAEpa_Etest_pu"   MAE of test energies per atom (physical units)
              - "MAE_Etrain_pu"    MAE of training energies (physical units)
              - "MAE_Etest_pu"     MAE of test energies (physical units)
              - "RMSE_Ftrain_pu"   RMSE of training forces (physical units)
              - "RMSE_Ftest_pu"    RMSE of test forces (physical units)
              - "MAE_Ftrain_pu"    MAE of training forces (physical units)
              - "MAE_Ftest_pu"     MAE of test forces (physical units)
            Default is `None`.
        file_out : str, optional
            The path to write the chosen weights to. Must be formatable with the atomic number.
            Default is "weights.{0:03d}.data".
        """
        if epoch is not None and minimum_criterion is not None:
            raise ValueError("Both `epoch` and `minimum_criterion` provided.")
        elif epoch is not None:
            # We already have the epoch to choose, so don't need to read performance.
            pass
        else:
            with open(join_paths(self.n2p2_directory, "learning-curve.out")) as f:
                lines = f.readlines()
            content = []
            for line in lines:
                if line.startswith("#    epoch"):
                    # Header for the rest of the table
                    headers = line[1:].split()
                elif not line.startswith("#"):
                    # Table content
                    content.append(line.split())

            if minimum_criterion is not None:
                try:
                    index = headers.index(minimum_criterion)
                except ValueError as e:
                    raise ValueError(
                        "`minimum_criterion={0}` not found in `learning-curve.out` headers: "
                        "{1}".format(minimum_criterion, headers)
                    ) from e
                values = [float(row[index]) for row in content]
                epoch, min_value = min(enumerate(values), key=lambda x: x[1])
                print(
                    "Epoch {0} chosen to minimise {1}".format(epoch, minimum_criterion)
                )
            else:
                # Default to the last epoch
                epoch = len(content) - 1

        for z in self.all_structures.atomic_number_list:
            src = join_paths(
                self.n2p2_directory, "weights.{0:03d}.{1:06d}.out".format(z, epoch)
            )
            dst = join_paths(self.n2p2_directory, file_out.format(z))
            copy(src=src, dst=dst)

    def write_extrapolations_lammps_script(
        self,
        ensembles: Tuple[str] = ("nve", "nvt", "npt"),
        temperatures: Tuple[int] = (340,),
        file_batch_template: str = "template.sh",
        file_batch_out: str = "lammps_extrapolations.sh",
    ):
        """
        Write batch script for using using LAMMPS to test the number of extrapolations that
        occur during simulation.

        Parameters
        ----------
        ensembles: tuple of str
            Contains all ensembles to run simulations with. Supported strings are "nve", "nvt"
            and "npt". Default is ("nve", "nvt", "npt").
        temperatures: tuple of int
            Contains all temperatures to run simulations at, in Kelvin. Only applies to "nvt"
            and "npt" ensembles. Default is (340,).
        file_batch_template: str, optional
            File location of template to use for batch scripts relative to
            `scripts_sub_directory`. Default is 'template.sh'.
        file_batch_out: str, optional
            File location to write the batch script to relative to
            `scripts_sub_directory`. Default is 'active_learning_nn.sh'.
        """
        with open(join_paths(self.scripts_directory, file_batch_template)) as f:
            batch_template_text = f.read()

        # Format SBATCH variables
        format_dict = {
            "job_name": "active_learning_NN",
            "nodes": 1,
            "job_array": "",
        }
        output_text = batch_template_text.format(**format_dict)

        # Setup
        output_text += "\nln -s {0} {1}/nnp".format(
            self.n2p2_directory, self.lammps_directory
        )

        for ensemble in ensembles:
            if "t" in ensemble:
                for t in temperatures:
                    output_text += "\nmpirun -np ${SLURM_NTASKS} "
                    output_text += (
                        "{0} -in {1}/md-{2}-t{3}.lmp > {1}/{2}-t{3}.log".format(
                            self.lammps_executable,
                            self.lammps_directory,
                            ensemble,
                            t,
                        )
                    )
                    # Create lammps input file
                    format_template_file(
                        template_file="{0}/md-{1}.lmp".format(
                            self.lammps_directory, ensemble
                        ),
                        formatted_file="{0}/md-{1}-t{2}.lmp".format(
                            self.lammps_directory, ensemble, t
                        ),
                        format_dict={"temp": str(t)},
                        format_shell_variables=True,
                    )
            else:
                output_text += "\nmpirun -np ${SLURM_NTASKS} "
                output_text += "{0} -in {1}/md-{2}.lmp > {1}/{2}.log".format(
                    self.lammps_executable,
                    self.lammps_directory,
                    ensemble,
                )

        output_text += "\nrm {0}/nnp".format(self.lammps_directory)

        with open(join_paths(self.scripts_directory, file_batch_out), "w") as f:
            f.write(output_text)
            print(
                "Batch script written to {}".format(
                    join_paths(self.scripts_directory, file_batch_out)
                )
            )

    def analyse_extrapolations(
        self,
        ensembles: Tuple[str] = ("nve", "nvt", "npt"),
        temperatures: Tuple[int] = (340,),
    ):
        """
        Read the number of successful steps taken in a LAMMPS simulation before stopping due to
        extrapolation warnings. This information is printed, and in the case of multiple
        temperature values the average number of steps is calculated.

        Parameters
        ----------
        ensembles: tuple of str
            Contains all ensembles to run simulations with. Supported strings are "nve", "nvt"
            and "npt". Default is ("nve", "nvt", "npt").
        temperatures: tuple of int
            Contains all temperatures to run simulations at, in Kelvin. Only applies to "nvt"
            and "npt" ensembles. Default is (340,).
        """
        timestep_data = {}

        for ensemble in ensembles:
            timestep_data[ensemble] = {}
            print(ensemble)
            print("Temp | T_step")
            if "t" in ensemble:
                for t in temperatures:
                    log_file = "{0}/{1}-t{2}.log".format(
                        self.lammps_directory,
                        ensemble,
                        t,
                    )
                    with open(log_file) as f:
                        lines = f.readlines()
                        line = lines.pop()
                        while (
                            "Too many extrapolation warnings" in line
                            or line.startswith("###")
                        ):
                            line = lines.pop()
                        timestep = int(line.split()[0])
                    timestep_data[ensemble][t] = timestep
                    print("{0:4d} | {1:5d}".format(t, timestep))
                timestep_data[ensemble]["mean"] = np.mean(
                    list(timestep_data[ensemble].values())
                )
                print(
                    "MEAN | {0:5d}\n".format(
                        int(round(timestep_data[ensemble]["mean"]))
                    )
                )
            else:
                timestep_data[ensemble] = {}
                log_file = "{0}/{1}.log".format(
                    self.lammps_directory,
                    ensemble,
                )
                with open(log_file) as f:
                    lines = f.readlines()
                    line = lines.pop()
                    while "Too many extrapolation warnings" in line or line.startswith(
                        "###"
                    ):
                        line = lines.pop()
                    timestep = int(line.split()[0])
                timestep_data[ensemble]["mean"] = timestep
                print("MEAN | {0:5d}\n".format(timestep_data[ensemble]["mean"]))

        return timestep_data

    def trim_dataset_separation(
        self,
        structure: Structure,
        data_file_in: str = "input.data",
        data_file_out: str = "input.data",
        data_file_backup: str = "input.data.minimum_separation_backup",
        data_file_unit: str = "Bohr",
    ):
        """
        Removes individual frames from `data_file_in` that do not meet the criteria on
        minimum separation set by `structure`. The frames that do satisfy the requirement
        are written to `data_file_out`. To prevent accidental overwrites, the original contents
        of `data_file_in` are also copied to `data_file_backup`.

        Parameters
        ----------
        structure: Structure
            The `Structure` represented in `data_file_in`, with requirements on the minimum
            separation of all constituent species with each other.
        data_file_in: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`,
            to read from. Default is "input.data".
        data_file_out: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`,
            to write to. Default is "input.data".
        data_file_backup: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`, to copy
            the original `data_file_in` to. Default is "input.data.minimum_separation_backup".
        data_file_unit: str, optional
            Length unit used in the data files, to ensure compatibility with the separation
            specified on `structure`.
        """
        remove_indices = []
        data = read_data_file(join_paths(self.n2p2_directory, data_file_in))
        for i, frame_data in enumerate(data):
            if not check_structure(
                lattice=frame_data[0] * self.units[data_file_unit],
                element=frame_data[1],
                position=frame_data[2] * self.units[data_file_unit],
                structure=structure,
            ):
                remove_indices.append(i)

        print(
            "Removing {} frames for having atoms within minimum separation."
            "".format(len(remove_indices))
        )

        remove_data(
            remove_indices,
            join_paths(self.n2p2_directory, data_file_in),
            join_paths(self.n2p2_directory, data_file_out),
            join_paths(self.n2p2_directory, data_file_backup),
        )

        return remove_indices

    def rebuild_dataset(
        self,
        atoms_per_frame: int,
        n_frames_to_select: int,
        n_frames_to_propose: int,
        n_frames_to_compare: int,
        select_extreme_frames: bool = True,
        starting_frame_indices: List[int] = None,
        criteria: Union[Literal["mean"], float] = "mean",
        seed: int = None,
        dtype: str = "float32",
        data_file_in: str = "input.data",
        data_file_out: str = "input.data",
        data_file_backup: str = "input.data.rebuild_backup",
    ) -> np.ndarray:
        """
        Taking the frames in `data_file_in` as the original dataset, reconstructs a new,
        smaller dataset and writes it to `data_file_out`.

        The selection of new structures is based on the separation of their symmetry functions,
        or fingerprints. A number of frames are proposed, and the euclidean distance of their
        fingerprint vectors to the vectors of already accepted frames are compared. Note that
        as multiple atomic environments are present in each frame, this difference is taken
        between all atoms of the same species. Using `criteria`, the proposed frame(s) that
        are most different from already accepted frames are added to the new dataset.

        Because of the high dimensionality involved, this is done in batches, with the number of
        frames to propose, compare against and select being configurable.

        Parameters
        ----------
        atoms_per_frame: int
            The number of atoms present in each frame.
        n_frames_to_select: int
            The number of frames to select from each batch.
        n_frames_to_propose: int
            The number of frames proposed at each batch.
        n_frames_to_compare: int
            The number of already accepted frames to compare against at each batch.
        select_extreme_frames: bool, optional
            If True, then frames containing a maximal or minimal symmetry function value is
            automatically selected and used in addition to `starting_frame_indices`. Note that
            if multiple frames minimise the same function, only one of these will be added.
            Default is True
        starting_frame_indices: list of int, optional
            If provided, these frames will be used as the initial set to compare against.
            If `None`, and `select_extreme_frames` is False, then `n_frames_compare` will be
            randomly selected instead. Default is `None`.
        criteria: float or "mean", optional
            If a float between 0 and 1, defines the quantile to take when comparing frames.
            For example, 1 would mean the maximum separation between two atomic environments
            is used to determine which frames to add, 0.5 would take the median separation.
            If "mean", then the mean of all environments is compared. Default is "mean".
        seed: int, optional
            The seed is used to randomly order the frames for selection. Default is `None`.
        dtype: str, optional
            numpy data type to use. Default is "float32".
        data_file_in: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`,
            to read from. Default is "input.data".
        data_file_out: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`,
            to write to. Default is "input.data".
        data_file_backup: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directory`, to copy
            the original `data_file_in` to. Default is "input.data.minimum_separation_backup".
        """
        np.random.seed(seed)
        t1 = time.time()
        atom_environments = read_atomenv(
            join_paths(self.n2p2_directory, "atomic-env.G"),
            self.elements,
            atoms_per_frame,
            dtype=dtype,
        )
        scaling_settings = read_nn_settings(
            join_paths(self.n2p2_directory, "input.nn"),
            requested_settings=[
                "scale_symmetry_functions",
                "scale_symmetry_functions_sigma",
                "scale_min_short",
                "scale_max_short",
                "center_symmetry_functions",
            ],
        )
        scaling = read_scaling(
            join_paths(self.n2p2_directory, "scaling.data"),
            self.elements,
        )
        t2 = time.time()
        print("Values read from file in {} s".format(t2 - t1))
        remove_indices = np.array([])
        frame_indices = np.random.permutation(len(list(atom_environments.values())[0]))
        range_indices = []

        if select_extreme_frames:
            # Calculate the min and max values for symmetry functions taking scaling into
            # account. Any frames that contain a minimal or maximal value are automatically
            # selected.
            if (
                "scale_symmetry_functions" in scaling_settings
                and "scale_symmetry_functions_sigma" in scaling_settings
            ):
                raise ValueError(
                    "Both scale_symmetry_functions and scale_symmetry_functions_sigma "
                    "were present in settings file."
                )
            elif "scale_symmetry_functions" in scaling_settings:
                if (
                    "scale_min_short" not in scaling_settings
                    or "scale_max_short" not in scaling_settings
                ):
                    raise ValueError(
                        "If scale_symmetry_functions is set, both scale_min_short and "
                        "scale_max_short must be present."
                    )
                for element_environments in atom_environments.values():
                    range_indices += list(
                        np.argmax(
                            np.any(
                                np.isclose(
                                    element_environments,
                                    float(scaling_settings["scale_min_short"]),
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )
                    range_indices += list(
                        np.argmax(
                            np.any(
                                np.isclose(
                                    element_environments,
                                    float(scaling_settings["scale_max_short"]),
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )
            elif "scale_symmetry_functions_sigma" in scaling_settings:
                for element, element_environments in atom_environments.items():
                    min_array = np.array(scaling[element]["min"])
                    max_array = np.array(scaling[element]["max"])
                    mean_array = np.array(scaling[element]["mean"])
                    sigma_array = np.array(scaling[element]["sigma"])
                    range_indices += list(
                        np.argmax(
                            np.any(
                                np.isclose(
                                    element_environments,
                                    (min_array - mean_array) / sigma_array,
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )
                    range_indices += list(
                        np.argmax(
                            np.any(
                                np.isclose(
                                    element_environments,
                                    (max_array - mean_array) / sigma_array,
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )
            else:
                for element, element_environments in atom_environments.items():
                    range_indices += list(
                        np.argmax(
                            np.any(
                                np.isclose(
                                    element_environments,
                                    np.array(scaling[element]["min"]),
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )
                    range_indices += list(
                        np.argmax(
                            np.any(
                                np.isclose(
                                    element_environments,
                                    np.array(scaling[element]["max"]),
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )

        # If starting_frame_indices provided, use those. Otherwise select starting frames
        # at random from the shuffled list of all frames.
        if starting_frame_indices is None and len(range_indices) == 0:
            selected_indices = frame_indices[:n_frames_to_compare]
            frame_indices = frame_indices[n_frames_to_compare:]
        elif starting_frame_indices is None:
            selected_indices = np.unique(range_indices)
            for starting_index in selected_indices:
                frame_indices = frame_indices[frame_indices != starting_index]
        else:
            selected_indices = np.unique(starting_frame_indices + range_indices)
            for starting_index in selected_indices:
                frame_indices = frame_indices[frame_indices != starting_index]
        print(
            "Starting rebuild with the following frames selected:\n{}\n"
            "".format(selected_indices)
        )

        while len(frame_indices) > 0:
            t3 = time.time()
            frames_compare = selected_indices[-n_frames_to_compare:]
            frames_proposed = frame_indices[:n_frames_to_propose]
            frame_indices = frame_indices[n_frames_to_propose:]
            total_metric = np.zeros(len(frames_proposed), dtype=dtype)

            for element_environments in atom_environments.values():
                environments_compare = element_environments[frames_compare].reshape(
                    element_environments[frames_compare].shape[0],
                    1,
                    element_environments[frames_compare].shape[1],
                    1,
                    element_environments[frames_compare].shape[2],
                )
                environments_proposed = element_environments[frames_proposed].reshape(
                    1,
                    element_environments[frames_proposed].shape[0],
                    1,
                    element_environments[frames_proposed].shape[1],
                    element_environments[frames_proposed].shape[2],
                )

                if isinstance(criteria, float):
                    if criteria > 1 or criteria < 0:
                        raise ValueError(
                            "`criteria` must be between 0 and 1, but was {}".format(
                                criteria
                            )
                        )
                    rmsd = (
                        np.mean(
                            ((environments_compare - environments_proposed)) ** 2,
                            axis=4,
                        )
                        ** 0.5
                    )
                    metric = np.quantile(rmsd, q=criteria, axis=(0, 2, 3))
                elif criteria == "mean":
                    metric = (
                        np.mean(
                            ((environments_compare - environments_proposed)) ** 2,
                            axis=(0, 2, 3, 4),
                        )
                        ** 0.5
                    )
                else:
                    raise ValueError(
                        "`criteria` must be a quantile (float) between 0 and 1 "
                        "or 'mean', but was {}".format(criteria)
                    )
                total_metric += metric

            max_separation_indicies = frames_proposed[
                np.argsort(total_metric)[-n_frames_to_select:]
            ]
            min_separation_indicies = frames_proposed[
                np.argsort(total_metric)[:-n_frames_to_select]
            ]
            selected_indices = np.concatenate(
                (selected_indices, max_separation_indicies)
            )
            remove_indices = np.concatenate((remove_indices, min_separation_indicies))
            print(
                "Proposed indices:\n{0}\n"
                "Difference metric summed over all elements:\n{1}\n".format(
                    frames_proposed, total_metric
                )
            )
            print("Selected indices:\n{}\n".format(max_separation_indicies))
            print("Time taken: {}\n".format(time.time() - t3))

        remove_data(
            remove_indices,
            join_paths(self.n2p2_directory, data_file_in),
            join_paths(self.n2p2_directory, data_file_out),
            join_paths(self.n2p2_directory, data_file_backup),
        )

        return selected_indices
