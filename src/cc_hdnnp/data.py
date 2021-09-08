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
from os.path import isdir, isfile, join
import re
from shutil import copy
from typing import List

from ase.io import read, write
from ase.units import create_units
import numpy as np

from cc_hdnnp.sfparamgen import SymFuncParamGenerator
from cc_hdnnp.structure import AllStructures


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
    ):
        self.units = create_units("2014")
        self.all_structures = structures
        # TODO convert to getters
        self.structure_names = structures.structure_dict.keys()
        self.elements = structures.element_list
        self.main_directory = main_directory
        self.lammps_executable = lammps_executable
        self.n2p2_bin = n2p2_bin
        self.scripts_directory = join(main_directory, scripts_sub_directory)
        self.n2p2_directory = join(main_directory, n2p2_sub_directory)
        self.active_learning_directory = join(
            main_directory, active_learning_sub_directory
        )

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
            join(self.main_directory, file_trajectory),
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
        self, file_structure: str, file_xyz: str, unit_in: str = "Bohr"
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
        xyz_path_split = join(self.main_directory, file_xyz).split("/")
        xyz_directory = "/".join(*xyz_path_split[:-2])
        if not isdir(xyz_directory):
            mkdir(xyz_directory)

        with open(join(self.active_learning_directory, file_structure)) as f:
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
                with open(join(self.main_directory, file_xyz.format(i)), "w") as f:
                    f.write(text)
                i += 1

    def remove_outliers(self, energy_threshold: float, force_threshold: float):
        """
        Following the "nnp-norm" command, "output.data" contains the normalised energies and
        forces. Read these and in "input.data", comment out the structures which have an energy
        or force above the specified threshold values.

        Note that scaling, normalising and pruning should be redone after this.

        Parameters
        ----------
        energy_threshold : float
            Structures which lie more than `energy_threshold` standard deviations away from the
            mean value will be removed from the dataset.
        energy_threshold : float
            Structures with a force component more than `energy_threshold` standard deviations
            away from the 0 will be removed from the dataset.
        """
        with open(join(self.n2p2_directory, "output.data")) as f:
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
                if abs(energy) > energy_threshold:
                    print(
                        "Structure {0} above threshold with an energy of {1}".format(
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

        if len(remove_indices) == 0:
            print("No outliers found")
            return

        print("{0} outliers found: {1}".format(len(remove_indices), remove_indices))
        copy(
            join(self.n2p2_directory, "input.data"),
            join(self.n2p2_directory, "input.data.outliers_included"),
        )
        with open(join(self.n2p2_directory, "input.data"), "r") as f:
            lines = f.readlines()
        with open(join(self.n2p2_directory, "input.data"), "w") as f:
            i = 0
            for line in lines:
                if i not in remove_indices:
                    f.write(line)

                if line.strip() == "end":
                    i += 1

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
                join(self.main_directory, file_xyz.format(i)),
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
                join(self.main_directory, file_xyz_in.format(i)), format=format
            )
            cell = frame.get_cell()
            frame.set_cell(cell * scale[i], scale_atoms=True)
            write(
                join(self.main_directory, file_xyz_out.format(i)),
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
            join(self.main_directory, file_input.format("template"))
        ) as f_template:
            input_template = f_template.read()

        with open(
            join(self.main_directory, file_batch.format("template"))
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
            with open(join(self.main_directory, file_xyz.format(i))) as f:
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
                    format_dict["file_xyz"] = join(
                        self.main_directory, file_xyz.format(i)
                    )
                    file_id = file_id_template.format(**format_dict)
                    file_input_formatted = join(
                        self.main_directory, file_input.format(file_id)
                    )
                    with open(file_input_formatted, "w") as f:
                        f.write(
                            input_template.format(
                                file_id=file_id,
                                structure_name=structure_name,
                                basis_set=basis_set,
                                potential=potential,
                                **format_dict
                            )
                        )

                    batch_scripts.append(
                        join(self.main_directory, file_batch.format(file_id))
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
                        join(self.main_directory, file_batch.format(file_id)), "w"
                    ) as f:
                        f.write(batch_text)

        bash_text = "sbatch "
        bash_text += "\nsbatch ".join(batch_scripts)
        with open(join(self.main_directory, file_bash), "w") as f:
            f.write(bash_text)

        return "bash {}".format(self.main_directory + file_bash)

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
                        join(self.main_directory, file_output.format(file_id))
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
                                **format_dict
                            )
                        )
                        msg_out += " | ".join(m_grid)
                        print(msg_out + " |")

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
            with open(join(self.main_directory, file_xyz.format(i))) as f:
                xyz_lines = f.readlines()
                n_atoms = int(xyz_lines[0].strip())
                header_list = xyz_lines[1].split('"')
                lattice_list = header_list[1].split()
                if n2p2_units["length"] != "Ang":
                    for j, lattice in enumerate(lattice_list):
                        lattice_list[j] = (
                            float(lattice) / self.units[n2p2_units["length"]]
                        )

            with open(join(self.main_directory, file_cp2k_forces.format(i))) as f:
                force_lines = f.readlines()

            with open(join(self.main_directory, file_cp2k_out.format(i))) as f:
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

        if isfile(join(self.n2p2_directory, file_n2p2_input)):
            with open(join(self.n2p2_directory, file_n2p2_input), "a") as f:
                f.write(text)
        else:
            with open(join(self.n2p2_directory, file_n2p2_input), "w") as f:
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
        file_nn_template: str = "n2p2/input.nn.template",
        file_nn: str = "n2p2/input.nn",
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

        if isfile(join(self.n2p2_directory, file_nn)):
            with open(join(self.n2p2_directory, file_nn), "a") as f:
                generator.write_settings_overview(fileobj=f)
                generator.write_parameter_strings(fileobj=f)
        else:
            with open(join(self.n2p2_directory, file_nn_template)) as f:
                template_text = f.read()
            with open(join(self.n2p2_directory, file_nn), "w") as f:
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
        with open(join(self.main_directory, file_batch_template)) as f:
            batch_template_text = f.read()

        format_dict = {"job_name": "n2p2_scale_prune", "nodes": nodes, "job_array": ""}
        output_text = batch_template_text.format(**format_dict)
        output_text += "\ncd {}".format(self.n2p2_directory)
        if normalise:
            output_text += "\nmpirun -np ${SLURM_NTASKS} " + join(
                self.n2p2_bin, "nnp-norm"
            )
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + join(self.n2p2_bin, "nnp-scaling")
            + " "
            + str(n_scaling_bins)
        )
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + join(self.n2p2_bin, "nnp-prune")
            + " range "
            + str(range_threshold)
        )
        output_text += "\nmv {0} {0}.unpruned".format(file_nn)
        output_text += "\nmv output-prune-range.nn {}".format(file_nn)
        output_text += (
            "\nmpirun -np ${SLURM_NTASKS} "
            + join(self.n2p2_bin, "nnp-scaling")
            + " "
            + str(n_scaling_bins)
        )

        with open(join(self.main_directory, file_prune), "w") as f:
            f.write(output_text)

        format_dict["job_name"] = "n2p2_train"
        output_text = batch_template_text.format(**format_dict)
        output_text += "\ncd {}".format(self.n2p2_directory)
        output_text += "\nmpirun -np ${SLURM_NTASKS} " + join(
            self.n2p2_bin, "nnp-train"
        )

        with open(join(self.main_directory, file_train), "w") as f:
            f.write(output_text)

    def write_lammps_data(
        self,
        file_xyz: str,
        file_data: str = "lammps/lammps.data",
        lammps_unit_style: str = "electron",
    ):
        """
        Reads data from xyz format and writes it for LAMMPS.

        Parameters
        ----------
        file_xyz : str
            Complete file name to read the xyz positions from.
        file_data: str, optional
            Complete file name to write the LAMMPS formatted positions to.
            Default is 'lammps/lammps.data'.
        lammps_unit_style: str, optional
            The LAMMPS unit system to use. The xyz is assumed to be in ASE
            default units (i.e. 'Ang' for length). Default is 'electron'.
        """
        format_in = "extxyz"
        format_out = "lammps-data"
        atoms = read(join(self.main_directory, file_xyz), format=format_in)
        write(
            join(self.main_directory, file_data),
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
        with open(join(self.main_directory, file_lammps_template)) as f:
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

        with open(join(self.main_directory, file_out), "w") as f:
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
        with open(join(self.scripts_directory, file_batch_template)) as f:
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

        with open(join(self.scripts_directory, file_batch_out), "w") as f:
            f.write(output_text)
            print(
                "Batch script written to {}".format(
                    join(self.scripts_directory, file_batch_out)
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
        with open(join(self.scripts_directory, file_batch_template)) as f:
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

        with open(join(self.scripts_directory, file_batch_out), "w") as f:
            f.write(output_text)
            print(
                "Batch script written to {}".format(
                    join(self.scripts_directory, file_batch_out)
                )
            )

    def choose_weights(self, epoch: int = None, minimum_criterion: str = None):
        """
        Copies the weights files for a particular epoch from "weights.XXX.YYYYYY.out" to
        "weights.XXX.data".

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
        """
        if epoch is not None and minimum_criterion is not None:
            raise ValueError("Both `epoch` and `minimum_criterion` provided.")
        elif epoch is not None:
            # We already have the epoch to choose, so don't need to read performance.
            pass
        else:
            with open(join(self.n2p2_directory, "learning-curve.out")) as f:
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
            else:
                # Default to the last epoch
                epoch = len(content) - 1

        for z in self.all_structures.atomic_number_list:
            src = join(
                self.n2p2_directory, "weights.{0:03d}.{1:06d}.out".format(z, epoch)
            )
            dst = join(self.n2p2_directory, "weights.{0:03d}.data".format(z))
            copy(src=src, dst=dst)
