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

TODO set up the import properly so we don't have to manually run this:

import sys
sys.path.append('/home/vol00/scarf860/cc_placement/CC_HDNNP/src')

"""

# from glob import glob
from os.path import isfile
import re
from typing import List

from ase.io import read, write
from ase.units import create_units
from file_manager import join_paths
from sfparamgen import SymFuncParamGenerator
from species import AllSpecies


class Data:
    """
    Holds information relevant to reading and writing data from and to file.

    Parameters
    ----------
    elements : AllSpecies
        Elements appearing in the system.
    data_directory : str
        All file names passed to other function will be appended to
        `data_directory`
    n2p2_bin : str
        File path to the n2p2 bin directory.
    """

    def __init__(
        self,
        elements: AllSpecies,
        data_directory: str,
        n2p2_bin: str,
        structure_names: List[str],
        active_learning_directory: str = None,
    ):
        #  n2p2_directory: str=None):
        self.units = create_units("2014")
        # TODO update relevant scripts with new attributes: n2p2_bin
        self.elements = elements
        self.data_directory = data_directory
        self.trajectory = None
        self.n2p2_bin = n2p2_bin
        # TODO unify this with
        self.structure_names = structure_names

        if active_learning_directory is not None:
            self.active_learning_directory = active_learning_directory
        else:
            self.active_learning_directory = join_paths(
                data_directory, "active_learning"
            )

        # if n2p2_directory is not None:
        #     self.n2p2_directory = active_learn_directory
        # else:
        #     self.n2p2_directory = data_directory + 'n2p2'

    def _min_n_config(self, n_provided: int):
        """
        Utility function to ensure we don't attempt to read/write more frames
        than were in the trajectory originally.

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
            File containing a trajectory of atom configurations
        format_in : str, optional
            File format for the ASE reader. Default is 'dlp-history'.
        unit_in: str, optional
            Length unit for the trajectory. Default is 'Ang'.

        Examples
        --------
        >>> from data import Data
        >>> d = Data('path/to/example_directory')
        >>> d.read_trajectory('example.history')
        >>> len(d.trajectory)
        101
        """
        trajectory = read(
            join_paths(self.data_directory, file_trajectory),
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

    def convert_n2p2_xyz(
        self, file_structure: str, file_xyz: str, unit_in: str = "Bohr"
    ):
        """
        Reads `file_structure` and writes it as a series of xyz files. By default length
        units of Bohr are assumed, so if this is not the case then `unit_in`
        should be specified to allow conversion. The xyz files will be in Ang.

        Parameters
        ----------
        file_structure : str
            File containing a series of structures.
        unit_in: str, optional
            Length unit for the trajectory. Default is 'Ang'.

        Returns
        -------
        Atoms or list of Atoms
        """
        with open(file_structure) as f:
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
                with open(file_xyz.format(i), "w") as f:
                    f.write(text)
                i += 1

    def write_xyz(self, file_xyz: str = "xyz/{}.xyz", unit_out: str = "Ang"):
        """
        Writes a loaded trajectory to file as a series of xyz files, optionally
        converting the units

        Parameters
        ----------
        file_xyz : str, optional
            Formatable file name to write the atomic co-ordinates to. Will be
            formatted with the frame number, so should contain `'{}'` as part
            of the string. Default is 'xyz/{}.xyz'.
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
                positions = frame.get_positions()
                frame.set_positions(positions / self.units[unit_out])

            write(
                join_paths(self.data_directory, file_xyz.format(i)),
                frame,
                format=format_out,
                columns=["symbols", "positions"],
            )

    def write_cp2k(
        self,
        file_bash: str = "scripts/all.sh",
        file_batch: str = "scripts/{}.sh",
        file_input: str = "cp2k_input/{}.inp",
        file_xyz: str = "xyz/{}.xyz",
        n_config: int = None,
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
            join_paths(self.data_directory, file_input.format("template"))
        ) as f_template:
            input_template = f_template.read()

        with open(
            join_paths(self.data_directory, file_batch.format("template"))
        ) as f_template:
            batch_template = f_template.read()

        n_config = self._min_n_config(n_config)
        file_id_template = "n_{i}"

        # TODO cast kwargs to lists if appropriate

        if "cutoff" in kwargs:
            file_id_template += "_cutoff_{cutoff}"
            cutoff_values = kwargs["cutoff"]
        else:
            cutoff_values = [None]

        if "relcutoff" in kwargs:
            file_id_template += "_relcutoff_{relcutoff}"
            relcutoff_values = kwargs["relcutoff"]
        else:
            relcutoff_values = [None]

        batch_scripts = []
        for i in range(n_config):
            format_dict = {"i": i}
            with open(join_paths(self.data_directory, file_xyz.format(i))) as f:
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
                        self.data_directory, file_xyz.format(i)
                    )
                    file_id = file_id_template.format(**format_dict)
                    with open(
                        join_paths(self.data_directory, file_input.format(file_id)), "w"
                    ) as f:
                        f.write(input_template.format(file_id=file_id, **format_dict))

                    batch_scripts.append(
                        join_paths(self.data_directory, file_batch.format(file_id))
                    )
                    with open(
                        join_paths(self.data_directory, file_batch.format(file_id)), "w"
                    ) as f:
                        f.write(batch_template.format(file_id=file_id, **format_dict))

        bash_text = "sbatch "
        bash_text += "\nsbatch ".join(batch_scripts)
        with open(join_paths(self.data_directory, file_bash)) as f:
            f.write(bash_text)

        return "bash {}".format(self.data_directory + file_bash)

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
            Arguments to be used in formatting the name of the cp2k output file(s). Each should be
            a tuple containing one or more values:
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

        msg = "| "
        if "cutoff" in kwargs:
            file_id_template += "_cutoff_{cutoff}"
            cutoff_values = kwargs["cutoff"]
            msg += "  {cutoff} | "
        else:
            cutoff_values = [None]

        if "relcutoff" in kwargs:
            file_id_template += "_relcutoff_{relcutoff}"
            relcutoff_values = kwargs["relcutoff"]
            msg += "  {relcutoff} | "
        else:
            relcutoff_values = [None]

        # Print table header:
        print(
            "| Cutoff |  Relative Cutoff | Processes "
            "| Energy                | t/step (s) | t total (s) "
            "| Grid 1 | Grid 2 | Grid 3 | Grid 4 |"
        )

        for i in range(n_config):
            for cutoff in cutoff_values:
                for relcutoff in relcutoff_values:
                    format_dict = {"i": i, "cutoff": cutoff, "relcutoff": relcutoff}

                    file_id = file_id_template.format(**format_dict)
                    with open(
                        join_paths(self.data_directory, file_output.format(file_id))
                    ) as f:
                        energy = None
                        m_grid = []
                        steps = None
                        total_time = None
                        for line in f.readlines():
                            if re.search("^ ENERGY|", line):
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
        file_cp2k_out: str = "cp2k_output/{}.log",
        file_cp2k_forces: str = "cp2k_output/{}-forces-1_0.xyz",
        file_xyz: str = "xyz/{}.xyz",
        file_n2p2_input: str = "n2p2/input.data",
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
            File name to write the n2p2 data to. Default is 'n2p2/input.data'.
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
        text = ""
        if n2p2_units is None:
            n2p2_units = {"length": "Bohr", "energy": "Ha", "force": "Ha / Bohr"}
        n = self._min_n_config(n_config)
        for i in range(n):
            with open(join_paths(self.data_directory, file_xyz.format(i))) as f:
                xyz_lines = f.readlines()
                n_atoms = int(xyz_lines[0].strip())
                header_list = xyz_lines[1].split('"')
                lattice_list = header_list[1].split()
                if n2p2_units["length"] != "Ang":
                    for j, lattice in enumerate(lattice_list):
                        lattice_list[j] = (
                            float(lattice) / self.units[n2p2_units["length"]]
                        )

            with open(join_paths(self.data_directory, file_cp2k_forces.format(i))) as f:
                force_lines = f.readlines()

            with open(join_paths(self.data_directory, file_cp2k_out.format(i))) as f:
                energy = None
                lines = f.readlines()
                for j, line in enumerate(lines):
                    if re.search("^ ENERGY|", line):
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
            # TODO allow for multiple structures (compatibility with active learning)
            text += "comment structure {}\n".format(self.structure_names[0])
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
                    for i in range(force):
                        force[i] = (
                            float(force)
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

        if isfile(join_paths(self.data_directory, file_n2p2_input)):
            with open(join_paths(self.data_directory, file_n2p2_input, "a")) as f:
                f.write(text)
        else:
            with open(join_paths(self.data_directory, file_n2p2_input), "w") as f:
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
        file_template: str = "n2p2/input.nn.template",
        file_nn: str = "n2p2/input.nn",
    ):
        """
        Based on `file_template`, write the input.nn file for n2p2 with
        symmetry functions generated using the provided arguments.

        Note that all distances (`r_cutoff`, `r_lower`, `r_upper`) should have
        the same units as the n2p2 `input.data` file (by default, Bohr).

        Parameters
        ----------
        r_cutoff: float
            The cutoff distance for the symmetry functions.
        type: {'radial', 'angular_narrow', 'angular_wide', 'weighted_radial', 'weighted_angular'}
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
        file_template : str, optional
            The file to read the general network architecture from (i.e.
            everything except the symmetry functions). Default is
            'n2p2/input.nn.template'.
        file_nn: str, optional
            The file to write the output to. If the file already exists, then
            it is appended to with the new symmetry functions. If it does not,
            then it is created and the text from `file_template` is written to
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

        generator = SymFuncParamGenerator(
            elements=self.elements.element_list, r_cutoff=r_cutoff
        )
        generator.symfunc_type = type
        generator.zetas = zetas

        generator.generate_radial_params(
            rule=rule,
            mode=mode,
            nb_param_pairs=n_pairs,
            r_lower=r_lower,
            r_upper=r_upper,
        )

        if isfile(join_paths(self.data_directory, file_nn)):
            with open(join_paths(self.data_directory, file_nn, "a")) as f:
                generator.write_settings_overview(fileobj=f)
                generator.write_parameter_strings(fileobj=f)
        else:
            with open(join_paths(self.data_directory, file_template)) as f:
                template_text = f.read()
            with open(join_paths(self.data_directory, file_nn), "w") as f:
                f.write(template_text)
                generator.write_settings_overview(fileobj=f)
                generator.write_parameter_strings(fileobj=f)

    def write_n2p2_scripts(
        self,
        n_scaling_bins: int = 500,
        range_threshold: float = 1e-4,
        nodes: int = 1,
        file_template: str = "scripts/template.bat",
        file_prune: str = "scripts/n2p2_prune.bat",
        file_train: str = "scripts/n2p2_train.bat",
        file_nn: str = "n2p2/input.nn",
    ):
        """
        Write batch script for scaling and pruning symmetry functions, and for
        training the network. Returns the command to submit the scripts.

        Parameters
        ----------
        n_scaling_bins: int, optional
            Number of bins for symmetry function histograms. Default is `500`.
        range_threshold: float, optional
            Symmetry functions with ranges below this will be "pruned" and not
            used in the training. Default is `1e-4`.
        nodes: int, optional
            Number of nodes to request for the batch job. Default is `1`.
        file_template: str, optional
            File location of template to use for batch scripts. Default is
            'scripts/template.bat'.
        file_prune: str, optional
            File location to write scaling and pruning batch script to.
            Default is 'scripts/n2p2_prune.bat'.
        file_train: str, optional
            File location to write training batch script to.
            Default is 'scripts/n2p2_train.bat'.
        file_nn: str, optional
            File location of n2p2 file defining the neural network. Default is
            n2p2/input.nn'.

        Returns
        -------
        str
            Command to run the scaling and pruning batch script.
        """
        with open(join_paths(self.data_directory, file_template)) as f:
            template_text = f.read()

        n2p2_directory = "/".join(file_nn.split("/")[:-1])
        output_text = template_text.format(job_name="n2p2_scale_prune", nodes=nodes)
        output_text += "\ncd {}".format(join_paths(self.data_directory, n2p2_directory))
        output_text += "\nsrun {0} {1}".format(
            join_paths(self.n2p2_bin, "nnp-scaling"), n_scaling_bins
        )
        output_text += "\nsrun {0} range {1}".format(
            join_paths(self.n2p2_bin, "nnp-prune"), range_threshold
        )
        output_text += "\nmv {0} {0}.unpruned".format(
            join_paths(self.data_directory, file_nn)
        )
        output_text += "\nmv output-prune-range.nn {}".format(
            join_paths(self.data_directory, file_nn)
        )

        with open(join_paths(self.data_directory, file_prune), "w") as f:
            f.write(output_text)

        # glob(self.data_directory + n2p2_directory + 'weights.*.*.out')
        # glob(self.data_directory + n2p2_directory + 'weights.*.000000.out')
        output_text = template_text.format(job_name="n2p2_train")
        output_text += "\ncd {}".format(join_paths(self.data_directory, n2p2_directory))
        output_text += "\nsrun {0}".format(join_paths(self.n2p2_bin, "nnp-train"))
        # TODO can't do this here as we don't have any files to list
        # Need to rename the weights files to use the network once trained
        # for atomic_number in self.elements.values():
        #     # File names always have the element as 3 digits, e.g. Hydrogen is
        #     # 001, so need to format accordingly
        #     atomic_number_str = str(atomic_number)
        #     atomic_number_str =  (3 - len(atomic_number_str)) * '0' + atomic_number_str
        #     weights = glob(self.data_directory + n2p2_directory +
        #                    'weights.{}.*.out'.format(atomic_number_str))
        #     weights.sort()
        #     # TODO currently assume the most recent set of weights, implement some more complex
        #     # criteria?
        #     output_text += '\ncp {0} weights.{1}.data'.format(weights[-1], atomic_number_str)

        with open(join_paths(self.data_directory, file_train), "w") as f:
            f.write(output_text)

        # return 'sbatch {0}\nsbatch {1}\n'.format(self.data_directory + file_prune,
        #                                          self.data_directory + file_train)

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
        atoms = read(join_paths(self.data_directory, file_xyz), format=format_in)
        write(
            join_paths(self.data_directory, file_data),
            atoms,
            format=format_out,
            units=lammps_unit_style,
        )

    def write_lammps_pair(
        self,
        r_cutoff: float,
        file_template: str = "lammps/template.lmp",
        file_out: str = "lammps/md.lmp",
        n2p2_directory: str = "n2p2",
        lammps_unit_style: str = "electron",
    ):
        """
        Writes the pair commands 'pair_style' and 'pair_coeff' to `file_out`,
        with the rest of the LAMMPS commands to be contained in `file_template`
        along with '{pair_commands}' to allow formatting that section to the
        following:

            pair_style nnp dir {n2p2_directory} showew no showewsum 10 resetew no maxew 100 emap \
                {elements_map}
            pair_coeff * * {r_cutoff}

        `r_cutoff` must be given in the LAMMPS units regardless of what was used for training.

        Parameters
        ----------
        r_cutoff: float
            The cutoff distance for the symmetry functions.
        file_template: str, optional
            Complete file name of the template for `file_out`. Should contain
            '{pair_commands}' to allow formatting. Default is
            'lammps/template.lmp'.
        file_template: str, optional
            Complete file name of the output file. Default is 'lammps/md.lmp'.
        n2p2_directory: str, optional
            Directory containing all the n2p2 files needed by the LAMMPS/n2p2
            interface. Default is 'n2p2'.
        lammps_unit_style: str, optional
            The LAMMPS unit system to use. Default is 'electron'.
        """
        with open(join_paths(self.data_directory, file_template)) as f:
            template_text = f.read()

        elements = self.elements.element_list
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
            raise ValueError("`lammps_unit_style={}` not recognised", lammps_unit_style)

        pair_coeff = "pair_coeff * * {}".format(r_cutoff)
        output_text = template_text.format(
            pair_commands=pair_style + "\n" + pair_coeff,
            lammps_unit_style=lammps_unit_style,
        )

        with open(join_paths(self.data_directory, file_out), "w") as f:
            f.write(output_text)
