"""
Central control object for high level user facing interactions. Mainly holds
information about the directory structure and executables so it can be passed
to other functions.
"""

from os import mkdir, remove
from os.path import exists, isdir, isfile, join as join_paths
import re
from shutil import copy, move
from typing import Dict, Iterable, List, Tuple, Union

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np

from cc_hdnnp.lammps_input import format_lammps_input
from cc_hdnnp.sfparamgen import SymFuncParamGenerator
from cc_hdnnp.slurm_input import format_slurm_input
from cc_hdnnp.structure import AllStructures, Structure
from .dataset import Dataset
from .file_readers import read_lammps_log
from .units import UNITS


class Controller:
    """
    Holds information about the directory structure and executables so it can be passed
    to other functions.

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
    n2p2_module_commands: List[str] = None
        Commands needed to load n2p2 for use in SLURM batch scripts. Each entry should be
        a separate command, e.g. ("module use ...", "module load ..."). Optional, default
        is None.
    cp2k_module_commands: List[str] = None
        Commands needed to load CP2K for use in SLURM batch scripts. Each entry should be
        a separate command, e.g. ("module use ...", "module load ..."). Optional, default
        is None.
    qe_module_commands: List[str] = None
        Commands needed to load QE for use in SLURM batch scripts. Each entry should be
        a separate command, e.g. ("module use ...", "module load ..."). Optional, default
        is None.
    scripts_sub_directory : str, optional
        Path for the directory to read/write scripts from/to, relative to the
        `main_directory`. Default is "scripts".
    n2p2_sub_directory : str or list of str, optional
        Path(s) for the directory to read/write n2p2 files from/to, relative to the
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
        n2p2_module_commands: List[str] = None,
        cp2k_module_commands: List[str] = None,
        qe_module_commands: List[str] = None,
        scripts_sub_directory: str = "scripts",
        n2p2_sub_directories: Union[str, List[str]] = "n2p2",
        active_learning_sub_directory: str = "active_learning",
        lammps_sub_directory: str = "lammps",
    ):
        self.all_structures = structures
        self.elements = structures.element_list
        self.main_directory = main_directory
        self.lammps_executable = lammps_executable
        self.n2p2_bin = n2p2_bin

        if n2p2_module_commands is not None:
            self.n2p2_module_commands = n2p2_module_commands
        else:
            self.n2p2_module_commands = []

        if cp2k_module_commands is not None:
            self.cp2k_module_commands = cp2k_module_commands
        else:
            self.cp2k_module_commands = []

        if qe_module_commands is not None:
            self.qe_module_commands = qe_module_commands
        else:
            self.qe_module_commands = []

        self.scripts_directory = join_paths(main_directory, scripts_sub_directory)
        if isinstance(n2p2_sub_directories, str):
            n2p2_sub_directories = [n2p2_sub_directories]
        self.n2p2_directories = [
            join_paths(main_directory, s) for s in n2p2_sub_directories
        ]
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
                frame.set_cell(cell * UNITS[unit_in])
                frame.set_positions(positions * UNITS[unit_in])

        self.trajectory = trajectory

    def convert_active_learning_to_xyz(
        self,
        file_n2p2_data: str,
        file_xyz: str,
        xyz_units: Dict[str, str] = None,
        single_output: bool = False,
    ):
        """
        Reads `file_n2p2_data` from `self.active_learning_directory` and writes it as a series
        of xyz files. By default length units of Bohr are assumed, so if this is not the case
        then `unit_in` should be specified to allow conversion.

        Parameters
        ----------
        file_n2p2_data : str
            File containing a series of structures in n2p2 format, relative to
            `self.active_learning_directory`.
        file_xyz : str
            Formatable file name to write to, relative to `self.main_directory`.
            If `single_output` is True, then a complete filename should be given instead.
        xyz_units: Dict[str, str] = None
            The units to write the extxyz file in. Should contain the key "length".
            Optional, default is None in which case Ang will be used.
        single_output: bool = False
            Whether to write all frames to a single file, or separate files formatted with
            the frame index.
        """
        xyz_path_split = join_paths(self.main_directory, file_xyz).split("/")
        xyz_directory = "/".join(xyz_path_split[:-1])
        if not isdir(xyz_directory):
            mkdir(xyz_directory)

        dataset = Dataset(
            data_file=join_paths(self.active_learning_directory, file_n2p2_data),
            all_structures=self.all_structures,
        )

        if xyz_units is None:
            xyz_units = {"length": "Ang", "energy": "eV"}
        dataset.change_units_all(new_units=xyz_units)

        if single_output:
            dataset.write(
                file_out=join_paths(self.main_directory, file_xyz), format="extxyz"
            )
        else:
            for i, frame in enumerate(dataset):
                write(
                    filename=join_paths(self.main_directory, file_xyz.format(i)),
                    images=frame,
                    format="extxyz",
                )

    def reduce_dataset_outliers(
        self,
        energy_threshold: Union[float, Tuple[float, float]],
        force_threshold: float,
        data_file_in: str = "input.data",
        data_file_out: str = "input.data",
        data_file_backup: str = "input.data.outliers_backup",
    ) -> List[int]:
        """
        Read `data_file_in` for energy and force values, and write only those Frames within
        the specified thresholds to `data_file_out`. `energy_threshold` and
        `force_threshold` should be in the same units as `data_file_in`.

        Note that scaling, normalising and pruning should be redone after this.

        Parameters
        ----------
        energy_threshold : float or tuple of float
            Structures which lie outside the range of `energy_threshold` will be removed from
            the dataset. The units depend on `data_file_in`. If a single value is given, the
            range taken is +- that value. Otherwise, the first and second entries are taken
            as the lower/upper bounds on energy.
        force_threshold : float
            Structures with a force component more than `force_threshold` will be removed from
            the dataset. The units depend on `data_file_in`.
        data_file_in: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directories`,
            to read from. Default is "input.data".
        data_file_out: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directories`,
            to write to. Default is "input.data".
        data_file_backup: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directories`, to copy
            the original `data_file_in` to. Default is "input.data.outliers_backup".

        Returns
        -------
        List[int]
            The list of frame indices that have been removed from the Dataset.
        """
        all_remove_indices = []
        if isinstance(energy_threshold, float):
            energy_threshold = (-energy_threshold, energy_threshold)

        for n2p2_directory in self.n2p2_directories:
            print("Removing outliers in {}".format(n2p2_directory))
            dataset = Dataset(
                data_file=join_paths(n2p2_directory, data_file_in),
                all_structures=self.all_structures,
            )
            conditions = dataset.check_threshold_all(
                energy_threshold=energy_threshold, force_threshold=force_threshold
            )

            dataset.write_data_file(
                file_out=join_paths(n2p2_directory, data_file_backup)
            )
            _, removed_indices = dataset.write_data_file(
                file_out=join_paths(n2p2_directory, data_file_out),
                conditions=conditions,
            )
            print(
                "Removing {} frames for having atoms outside of threshold."
                "".format(len(removed_indices))
            )
            all_remove_indices.append(removed_indices)

        return all_remove_indices

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
                frame.set_cell(cell / UNITS[unit_out])
                frame.set_positions(positions / UNITS[unit_out])

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
        basis_set_directory: str,
        potential_directory: str,
        basis_set_dict: Dict[str, str],
        potential_dict: Dict[str, str],
        file_bash: str = "all.sh",
        file_batch: str = "{}.sh",
        file_input: str = "cp2k_input/{}.inp",
        file_xyz: str = "xyz/{}.xyz",
        n_config: int = None,
        **kwargs,
    ) -> str:
        """
        Writes .inp files and batch scripts for running cp2k from `n_config`
        .xyz files. Can set supported settings using `**kwargs`, in which case
        template file(s) will be formatted to contain the values provided. Note
        that the .xyz files should be in 'Ang'. Returns the command for running
        all cp2k files.

        Can also use `**kwargs` to set optional arguments for the SLURM batch script.

        Parameters
        ----------
        structure_name : str
            Name of the structure, used as part of the CP2K project name which in turn
            determines file names.
        basis_set_directory : str
            Filepath to the CP2K basis set to use.
        potential_directory : str
            Filepath to the CP2K potential to use.
        basis_set_dict: Dict[str, str]
            Keys are chemical symbol of the element, values are the files within the
            `basis_set_directory` to use.
        potential_dict: Dict[str, str]
            Keys are chemical symbol of the element, values are the files within the
            `potential_directory` to use.
        file_bash : str, optional
            File name to write a utility script which submits all of the batch
            scripts created by this function. Default is 'all.sh'.
        file_batch : str, optional
            Formatable file name to write the batch scripts to. Will be
            formatted with the frame number and any other `**kwargs`, so should
            contain '{}' as part of the string. There should already be a
            template of this file with 'template' instead of '{}' containing
            the details of the system that will remain constant across all
            frames and so do not need formatting. Default is '{}.sh'.
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
            Can also be used to set optional str arguments for the batch script:
              - constraint
              - nodes
              - ntasks_per_node
              - time
              - out
              - account
              - reservation
              - exclusive
              - commands

        Returns
        -------
        str
            Command to run `file_bash`, which in turn will submit all batch scripts.
        """
        # Read the template for the CP2K file
        with open(
            join_paths(self.main_directory, file_input.format("template"))
        ) as f_template:
            input_template = f_template.read()

        pseudos_text = ""
        for e in self.elements:
            pseudos_text += (
                f"    &KIND {e}\n"
                f"      BASIS_SET {basis_set_dict[e]}\n"
                f"      POTENTIAL {potential_dict[e]}\n"
                "    &END\n"
            )

        n_config = self._min_n_config(n_config)
        file_id_template = "n_{i}"

        if "cutoff" in kwargs:
            file_id_template += "_cutoff_{cutoff}"
            cutoff_values = kwargs.pop("cutoff")
            if isinstance(cutoff_values, float) or isinstance(cutoff_values, int):
                cutoff_values = [cutoff_values]
        else:
            cutoff_values = [None]

        if "relcutoff" in kwargs:
            file_id_template += "_relcutoff_{relcutoff}"
            relcutoff_values = kwargs.pop("relcutoff")
            if isinstance(relcutoff_values, float) or isinstance(relcutoff_values, int):
                relcutoff_values = [relcutoff_values]
        else:
            relcutoff_values = [None]

        bash_text = ""
        for cutoff in cutoff_values:
            for relcutoff in relcutoff_values:
                for i in range(n_config):
                    # Extract the lattice vectors from the xyz file
                    file_xyz_i = join_paths(self.main_directory, file_xyz.format(i))
                    with open(file_xyz_i) as f:
                        header_line = f.readlines()[1]
                        lattice_string = header_line.split('"')[1]
                        lattice_list = lattice_string.split()
                        cell_x = " ".join(lattice_list[0:3])
                        cell_y = " ".join(lattice_list[3:6])
                        cell_z = " ".join(lattice_list[6:9])

                    file_id = file_id_template.format(
                        i=i, cutoff=cutoff, relcutoff=relcutoff
                    )
                    file_input_i = join_paths(
                        self.main_directory, file_input.format(file_id)
                    )
                    with open(file_input_i, "w") as f:
                        f.write(
                            input_template.format(
                                file_id=file_id,
                                structure_name=structure_name,
                                basis_set=basis_set_directory,
                                potential=potential_directory,
                                pseudos_text=pseudos_text,
                                cell_x=cell_x,
                                cell_y=cell_y,
                                cell_z=cell_z,
                                cutoff=cutoff,
                                relcutoff=relcutoff,
                                file_xyz=file_xyz_i,
                            )
                        )

                    file_batch_i = join_paths(
                        self.scripts_directory,
                        file_batch.format(f"cp2k_cutoff{cutoff}_relcutoff{relcutoff}"),
                    )

                    commands = kwargs.pop("commands", [])
                    run_command = (
                        "mpirun -np ${SLURM_NTASKS} cp2k.popt "
                        f"{file_input_i} &> ../cp2k_output/{structure_name}_{file_id}.log"
                    )
                    move_command = (
                        f"mv {structure_name}_{file_id}-forces-1_0.xyz "
                        f"../cp2k_output/{structure_name}_{file_id}-forces-1_0.xyz"
                    )
                    format_slurm_input(
                        formatted_file=file_batch_i,
                        commands=[
                            *self.cp2k_module_commands,
                            run_command,
                            move_command,
                            *commands,
                        ],
                        job_name="CP2K",
                        array=f"0-{n_config - 1}",
                        **kwargs,
                    )
                    bash_text += f"sbatch {file_batch_i}\n"

        with open(join_paths(self.scripts_directory, file_bash), "w") as f:
            f.write(bash_text)

        return "bash {}".format(file_bash)

    def write_qe_input(
        self,
        atoms: Atoms,
        frame_directory: str,
        structure: Structure,
        pseudos: Dict[str, str],
        **kwargs,
    ):
        """
        Writes the input file for Quantum Espresso to `frame_directory` for the `atoms`
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
        pseudos: Dict[str, str]
            Keys are chemical symbol of the element, values are the files within the pseudos
            directory to use.
        **kwargs: Any
            Additional keyword arguments can be used to change the QE settings from their
            default values. Possible settings and their default values are:
                'ibrav'             : 14,
                'calculation'       :'scf',
                'conv_thr'          : 1.0e-8,
                'diago_david_ndim'  : 4,
                'mixing_beta'       : 0.25,
                'startingwfc'       : 'atomic+random',
                'startingpot'       : 'atomic',
                'nbnd'              : 400,
                'ecutwfc'           : 80,
                'ecutrho'           : 480,
                'input_dft'         : 'VDW-DF2-B86R',
                'occupations'       : 'fixed',
                'degauss'           : 0.08,
                'smearing'          : 'm-p',
                'tstress'           : False,
                'tprnfor'           : True,
                'verbosity'         : "high",
                'outdir'            : './{}'.format(structure.name),
                'pseudo_dir'        : "../pseudos/",
                'disk_io'           : "none",
                'restart_mode'      : 'restart',
        """
        options = {
            "ibrav": 14,
            "calculation": "scf",
            "conv_thr": 1.0e-8,
            "diago_david_ndim": 4,
            "mixing_beta": 0.25,
            "startingwfc": "atomic+random",
            "startingpot": "atomic",
            "nbnd": 400,
            "ecutwfc": 80,
            "ecutrho": 480,
            "input_dft": "VDW-DF2-B86R",
            "occupations": "fixed",
            "degauss": 0.08,
            "smearing": "m-p",
            "tstress": False,
            "tprnfor": True,
            "verbosity": "high",
            "outdir": "./{}".format(structure.name),
            "pseudo_dir": "../pseudos/",
            "disk_io": "none",
            "restart_mode": "restart",
        }

        for key, value in kwargs.items():
            if key not in options:
                raise ValueError(
                    "Key value pair {}: {} passed as **kwarg not one of the recognised "
                    "options: {}".format(key, value, list(options))
                )
            options[key] = value

        write(
            filename=join_paths(frame_directory, "{}.in".format(structure.name)),
            images=atoms,
            input_data=options,
            pseudopotentials=pseudos,
            format="espresso-in",
        )

    def write_qe_pp(self, frame_directory: str, structure: Structure):
        """
        Write "pp.in" input script for Quantum Espresso plotting.

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

    def write_qe_slurm(self, frame_directory: str, structure: Structure, **kwargs):
        """
        Writes the batch script for running all stages of Quantum Espresso.

        Parameters
        ----------
        frame_directory: str
            Complete path to the directory in which to write the QE input file.
        structure: Structure
            The `Structure` being simulated. Will be used to determine the file name and
            get relevant elements.
        **kwargs:
            Can be used to set optional str arguments for the batch script:
              - constraint
              - nodes
              - ntasks_per_node
              - time
              - out
              - account
              - reservation
              - exclusive
              - commands
        """
        commands = self.qe_module_commands
        commands += [
            "n=$SLURM_NTASKS",
            "export OMP_NUM_THREADS=1",
            f"cd {frame_directory}",
            f"mpirun -n $n pw.x  -i  {structure.name}.in > {structure.name}.log",
            f"mpirun -n $n pp.x  <  pp.in > {structure.name}-pp.log",
            f"/home/vol02/scarf562/bin/bader {structure.name}.cube > bader.log",
            f"rm -f {structure.name}.cube",
            f"rm -rf ./{structure.name}/pwscf.save",
            "rm -f tmp.pp",
        ]
        commands += kwargs.pop("commands", [])

        formatted_file = join_paths(frame_directory, "qe.slurm")
        format_slurm_input(
            formatted_file=join_paths(frame_directory, "qe.slurm"),
            commands=commands,
            job_name=f"QE_{structure.name}",
            **kwargs,
        )

        return f"sbatch {formatted_file}"

    def prepare_qe(
        self,
        temperatures: List[int],
        pressures: List[int],
        structure: Structure,
        pseudos: Dict[str, str],
        qe_directory: str = "qe",
        selection: Tuple[int, int] = (0, 1),
        **kwargs,
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
        pseudos: Dict[str]
            File names of the pseudo potentials to use, found within `pseudos` directory and
            keyed by atomic number.
        selection: tuple of int
            Allows for subsampling of the trajectory files. First entry is the index of the
            first frame to use, second allows for every nth frame to be selected.
        **kwargs:
            Can be used to set optional str arguments for the batch script:
              - constraint
              - nodes
              - ntasks_per_node
              - time
              - out
              - account
              - reservation
              - exclusive
              - commands
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
                        submit_all_text += self.write_qe_slurm(
                            folder, structure, **kwargs
                        )
                        submit_all_text += "\n"

        if submit_all_text:
            with open(join_paths(self.scripts_directory, "qe_all.sh"), "w") as f:
                f.write(submit_all_text)

    def print_cp2k_table(
        self,
        structure_name: str,
        file_output: str = "cp2k_output/{}.log",
        n_config: int = None,
        n_mpi: int = 1,
        **kwargs,
    ):
        """
        Print the final energy, time taken, and grid allocation for given cp2k
        settings. Formatted for a .md table.

        Parameters
        ----------
        structure_name : str
            Name of the structure, used as part of the CP2K project name which in turn
            determines file names.
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
        """
        n_config = self._min_n_config(n_config)
        file_id_template = structure_name + "_n_{i}"

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
        temperatures: Iterable[int],
        pressures: Iterable[int],
        selection: Tuple[int, int] = (0, 1),
        qe_directory: str = "qe",
        file_qe_log: str = "T{temperature}-p{pressure}-{index}/{structure_name}.log",
        file_qe_charges: str = "T{temperature}-p{pressure}-{index}/ACF.dat",
        file_xyz: str = "{structure_name}-T{temperature}-p{pressure}.xyz",
        file_n2p2_input: str = "input.data",
        n2p2_units: Dict[str, str] = None,
    ):
        """
        Read the Quantum Espresso output files, then format the information for n2p2 and write
        to file.

        Parameters
        ----------
        structure_name: str
            The name of the Structure in question.
        temperatures: Iterable[int]
            All temperatures that Quantum Espresso was run for.
        pressures: Iterable[int]
            All pressures that Quantum Espresso was run for.
        selection: tuple of int
            Allows for subsampling of the trajectory files. First entry is the index of the
            first frame to use, second allows for every nth frame to be selected.
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
            File path to write the n2p2 data to, relative to `self.n2p2_directories`.
            Default is "input.data".
        n2p2_units: Dict[str, str] = None
            The units to use for n2p2. No specific units are required, however
            they should be consistent (i.e. positional data and symmetry
            functions can use 'Ang' or 'Bohr' provided both do). Default is `None`, which will
            lead to `{'length': 'Bohr', 'energy': 'Ha'}` being used.
        """
        if structure_name not in self.all_structures.keys():
            raise ValueError(
                "`structure_name` {} not recognized".format(structure_name)
            )

        if n2p2_units is None:
            n2p2_units = {"length": "Bohr", "energy": "Ha"}

        for temperature in temperatures:
            for pressure in pressures:
                dataset = Dataset(
                    data_file=join_paths(
                        self.main_directory,
                        qe_directory,
                        file_xyz.format(
                            structure_name=structure_name,
                            temperature=temperature,
                            pressure=pressure,
                        ),
                    ),
                    all_structures=self.all_structures,
                    format="extxyz",
                )
                dataset.change_units_all(new_units=n2p2_units)
                remove_indices = []
                for index, frame in enumerate(dataset):
                    if index >= selection[0] and index % selection[1] == 0:
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
                            with open(full_filepath) as f:
                                lines = f.readlines()
                            energy = None
                            forces = None
                            for i, line in enumerate(lines):
                                if line.startswith("!    total energy"):
                                    energy = (
                                        float(line.split()[4])
                                        * UNITS[line.split()[5]]
                                        / UNITS[n2p2_units["energy"]]
                                    )
                                elif line.startswith("     Forces acting on atoms"):
                                    forces = [
                                        force_line.split()[-3:]
                                        for force_line in lines[
                                            i + 2 : i + 2 + len(frame)
                                        ]
                                    ]
                                    forces = np.array(forces).astype(float)
                                    qe_force_unit = line.split()[-1][:-2]
                                    forces *= UNITS[qe_force_unit.split("/")[0]]
                                    forces /= UNITS[qe_force_unit.split("/")[1]]
                                    forces /= UNITS[n2p2_units["energy"]]
                                    forces *= UNITS[n2p2_units["length"]]

                            if energy is None:
                                print(
                                    "{} did not complete no energy found, skipping".format(
                                        full_filepath
                                    )
                                )
                                remove_indices.append(index)
                                continue

                            if forces is None:
                                print(
                                    "{} did not complete no forces found, skipping".format(
                                        full_filepath
                                    )
                                )
                                remove_indices.append(index)

                            frame.energy = energy
                            frame.forces = forces
                            if exists(charge_filepath):
                                charges = self.read_charges_qe(
                                    charge_filepath, len(frame)
                                )
                                relative_charges = []
                                structure = self.all_structures[structure_name]
                                for i, symbol in enumerate(frame.symbols):
                                    valence = structure.get_species(symbol).valence
                                    if (
                                        charges is None
                                        or len(frame) != len(charges)
                                        or valence is None
                                    ):
                                        relative_charges.append(0.0)
                                    else:
                                        relative_charges.append(charges[i])
                                frame.set_initial_charges(charges=relative_charges)

                        else:
                            print("{} not found, skipping".format(full_filepath))
                            remove_indices.append(index)
                    else:
                        # Remove structures if they do not match selection criteria
                        remove_indices.append(index)

        if len(dataset) == len(remove_indices):
            raise IOError("No files found.")

        for n2p2_directory in self.n2p2_directories:
            dataset.write_data_file(
                file_out=join_paths(n2p2_directory, file_n2p2_input),
                conditions=(i not in remove_indices for i, _ in enumerate(dataset)),
                append=True,
            )

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
            File name to write the n2p2 data to relative to `self.n2p2_directories`.
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
        if structure_name not in self.all_structures.keys():
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
                        lattice_list[j] = float(lattice) / UNITS[n2p2_units["length"]]

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
                energy = float(energy) * UNITS["Ha"] / UNITS[n2p2_units["energy"]]

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
                    atom_xyz[k] = float(position) / UNITS[n2p2_units["length"]]

                force = force_lines[j + 4].split()[-3:]
                if n2p2_units["force"] != "Ha / Bohr":
                    # cp2k output is in Ha / Bohr
                    for i in range(len(force)):
                        force[i] = (
                            float(force[i])
                            * UNITS["Ha"]
                            / UNITS["Bohr"]
                            / UNITS[n2p2_units["energy"]]
                        )

                charge = charge_lines[j].split()[-1]
                text += "atom {1} {2} {3} {0} {4} 0.0 {5} {6} {7}\n".format(
                    *atom_xyz + [charge] + force
                )

            text += "energy {}\n".format(energy)
            text += "charge {}\n".format(total_charge)
            text += "end\n"

        for n2p2_directory in self.n2p2_directories:
            if isfile(join_paths(n2p2_directory, file_n2p2_input)):
                with open(join_paths(n2p2_directory, file_n2p2_input), "a") as f:
                    f.write(text)
            else:
                with open(join_paths(n2p2_directory, file_n2p2_input), "w") as f:
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
        relative to `self.n2p2_directories`.

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
            'n2p2/input.nn'.
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

        for n2p2_directory in self.n2p2_directories:
            if isfile(join_paths(n2p2_directory, file_nn)):
                with open(join_paths(n2p2_directory, file_nn), "a") as f:
                    generator.write_settings_overview(fileobj=f)
                    generator.write_parameter_strings(fileobj=f)
            else:
                with open(join_paths(n2p2_directory, file_nn_template)) as f:
                    template_text = f.read()
                with open(join_paths(n2p2_directory, file_nn), "w") as f:
                    f.write(template_text)
                    generator.write_settings_overview(fileobj=f)
                    generator.write_parameter_strings(fileobj=f)

    def write_n2p2_scripts(
        self,
        normalise: bool = False,
        n_scaling_bins: int = 500,
        range_threshold: float = 1e-4,
        file_prepare: str = "n2p2_prepare.sh",
        file_train: str = "n2p2_train.sh",
        file_nn: str = "input.nn",
        **kwargs,
    ):
        """
        Write batch script for scaling and pruning symmetry functions, and for
        training the network. Returns the command to submit the scripts.

        Can also use `**kwargs` to set optional arguments for the SLURM batch script.

        Parameters
        ----------
        normalise: bool, optional
            Whether to apply normalisation to the network. Default is `False`.
        n_scaling_bins: int, optional
            Number of bins for symmetry function histograms. Default is `500`.
        range_threshold: float, optional
            Symmetry functions with ranges below this will be "pruned" and not
            used in the training. Default is `1e-4`.
        file_prepare: str, optional
            File location to write scaling and pruning batch script to.
            Default is 'n2p2_prepare.sh'.
        file_train: str, optional
            File location to write training batch script to.
            Default is 'n2p2_train.sh'.
        file_nn: str, optional
            File location of n2p2 file defining the neural network. Default is
            "input.nn".
        **kwargs:
            Used to set optional str arguments for the batch script:
              - constraint
              - nodes
              - ntasks_per_node
              - time
              - out
              - account
              - reservation
              - exclusive
              - commands
        """

        n2p2_directory_str = " ".join((f"'{d}'" for d in self.n2p2_directories))
        common_commands = self.n2p2_module_commands + [
            f"n2p2_directories=({n2p2_directory_str})",
            "cd ${n2p2_directories[${SLURM_ARRAY_TASK_ID}]}",
        ]

        prepare_commands = common_commands.copy()
        if normalise:
            prepare_commands += [
                "mpirun -np ${SLURM_NTASKS} " + join_paths(self.n2p2_bin, "nnp-norm")
            ]

        prepare_commands += [
            (
                "mpirun -np ${SLURM_NTASKS} "
                + join_paths(self.n2p2_bin, "nnp-scaling")
                + f" {n_scaling_bins}"
            ),
            (
                "mpirun -np ${SLURM_NTASKS} "
                + join_paths(self.n2p2_bin, "nnp-prune")
                + f" range {range_threshold}"
            ),
            f"mv {file_nn} {file_nn}.unpruned",
            f"mv output-prune-range.nn {file_nn}",
            (
                "mpirun -np ${SLURM_NTASKS} "
                + join_paths(self.n2p2_bin, "nnp-scaling")
                + f" {n_scaling_bins}"
            ),
        ]

        format_slurm_input(
            formatted_file=join_paths(self.scripts_directory, file_prepare),
            commands=prepare_commands,
            job_name="n2p2_prepare",
            array=f"0-{len(self.n2p2_directories) - 1}",
            **kwargs,
        )

        train_commands = common_commands.copy()
        train_commands += [
            "mpirun -np ${SLURM_NTASKS} " + join_paths(self.n2p2_bin, "nnp-train")
        ]

        format_slurm_input(
            formatted_file=join_paths(self.scripts_directory, file_train),
            commands=train_commands,
            job_name="n2p2_train",
            array=f"0-{len(self.n2p2_directories) - 1}",
            **kwargs,
        )

        return f"sbatch {file_prepare}; sbatch {file_train}"

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

    def _write_active_learning_lammps_script(
        self,
        n_simulations: int,
        file_batch_out: str = "active_learning_lammps.sh",
        **kwargs,
    ):
        """
        Write batch script for using LAMMPS to generate configurations for active learning.

        Can also use `**kwargs` to set optional arguments for the SLURM batch script.

        Parameters
        ----------
        n_simulations: int
            Number of simulations required. This is used to set the upper limit on the SLURM
            job array.
        file_batch_out: str, optional
            File location to write the batch script relative to `scripts_sub_directory`.
            Default is 'scripts/active_learning_lammps.sh'.
        **kwargs:
            Used to set optional str arguments for the batch script:
              - constraint
              - nodes
              - ntasks_per_node
              - time
              - out
              - account
              - reservation
              - exclusive
              - commands
        """
        commands = self.n2p2_module_commands + [
            "ulimit -s unlimited",
            (
                "path=$(sed -n ${SLURM_ARRAY_TASK_ID}p ${SLURM_SUBMIT_DIR}/"
                f"{self.active_learning_directory}/joblist_mode1.dat)"
            ),
            "dir=$(date '+%Y%m%d_%H%M%S_%N')",
            "mkdir -p /scratch/$(whoami)/${dir}",
            (
                "rsync -a ${SLURM_SUBMIT_DIR}/"
                f"{self.active_learning_directory}"
                "/mode1/${path}/* /scratch/$(whoami)/${dir}/"
            ),
            "cd /scratch/$(whoami)/${dir}",
            (
                "mpirun -np ${SLURM_NTASKS} "
                f"{self.lammps_executable} -in input.lammps -screen none"
            ),
            "rm -r /scratch/$(whoami)/${dir}/RuNNer",
            (
                "rsync -a /scratch/$(whoami)/${dir}/* ${SLURM_SUBMIT_DIR}/"
                f"{self.active_learning_directory}"
                "/mode1/${path}/"
            ),
            "rm -r /scratch/$(whoami)/${dir}",
        ]

        format_slurm_input(
            formatted_file=join_paths(self.scripts_directory, file_batch_out),
            commands=commands,
            job_name="active_learning_LAMMPS",
            array=f"1-{n_simulations}",
            **kwargs,
        )

        return f"sbatch {file_batch_out}"

    def _write_active_learning_nn_script(
        self,
        file_batch_out: str = "active_learning_nn.sh",
        **kwargs,
    ):
        """
        Write batch script for using the neural network to calculate energies for
        configurations as part of the active learning.

        Can also use `**kwargs` to set optional arguments for the SLURM batch script.

        Parameters
        ----------
        file_batch_out: str, optional
            File location to write the batch script to relative to
            `scripts_sub_directory`. Default is 'active_learning_nn.sh'.
        **kwargs:
            Used to set optional str arguments for the batch script:
              - constraint
              - nodes
              - ntasks_per_node
              - time
              - out
              - account
              - reservation
              - exclusive
              - commands
        """
        # Create directories if they don't exist
        mode2 = "{}/mode2".format(self.active_learning_directory)
        HDNNP_1 = "{}/mode2/HDNNP_1".format(self.active_learning_directory)
        HDNNP_2 = "{}/mode2/HDNNP_2".format(self.active_learning_directory)
        for dir in (mode2, HDNNP_1, HDNNP_2):
            if not isdir(dir):
                mkdir(dir)

        # Write copies of data to directories
        dataset = Dataset(
            data_file="{}/input.data-new".format(self.active_learning_directory),
            all_structures=self.all_structures,
        )
        dataset.write_data_file(
            file_out="{}/mode2/HDNNP_1/input.data".format(
                self.active_learning_directory
            )
        )
        dataset.write_data_file(
            file_out="{}/mode2/HDNNP_2/input.data".format(
                self.active_learning_directory
            )
        )

        n2p2_directory_str = " ".join((f"'{d}'" for d in self.n2p2_directories))
        commands = self.n2p2_module_commands + [
            f"n2p2_directories=({n2p2_directory_str})",
            (
                "cp ${n2p2_directories[${SLURM_ARRAY_TASK_ID} - 1]}/input.nn "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}"
            ),
            (
                "cp ${n2p2_directories[${SLURM_ARRAY_TASK_ID} - 1]}/scaling.data "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}"
            ),
            (
                "cp ${n2p2_directories[${SLURM_ARRAY_TASK_ID} - 1]}/weights.*.data "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}"
            ),
            (
                "sed -i s/'.*test_fraction.*'/'test_fraction 0.0'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                "sed -i s/'epochs.*'/'epochs 0'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                "sed -i s/'.*use_old_weights_short'/'use_old_weights_short'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                "sed -i s/'.*use_short_forces'/'use_short_forces'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                "sed -i s/'.*write_trainpoints'/'write_trainpoints'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                "sed -i s/'.*write_trainforces'/'write_trainforces'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                "sed -i s/'.*precondition_weights'/'#precondition_weights'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                "sed -i s/'.*nguyen_widrow_weights_short'/'#nguyen_widrow_weights_short'/g "
                f"{self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}/input.nn"
            ),
            (
                f"cd {self.active_learning_directory}/mode2"
                "/HDNNP_${SLURM_ARRAY_TASK_ID}"
            ),
            "mpirun -np ${SLURM_NTASKS} " + f"{self.n2p2_bin}/nnp-train > mode_2.out",
        ]

        format_slurm_input(
            formatted_file=join_paths(self.scripts_directory, file_batch_out),
            commands=commands,
            job_name="active_learning_NN",
            array="1-2",
            **kwargs,
        )

        return f"sbatch {file_batch_out}"

    def choose_weights(
        self,
        n2p2_directory_index: int = 0,
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
        n2p2_directory_index: str, optional
            The index of the directory within `self.n2p2_directories` containing the weights
            files. Default is 0.
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
            with open(
                join_paths(
                    self.n2p2_directories[n2p2_directory_index], "learning-curve.out"
                )
            ) as f:
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
                self.n2p2_directories[n2p2_directory_index],
                "weights.{0:03d}.{1:06d}.out".format(z, epoch),
            )
            dst = join_paths(
                self.n2p2_directories[n2p2_directory_index], file_out.format(z)
            )
            copy(src=src, dst=dst)

    def write_extrapolations_lammps_script(
        self,
        n2p2_directory_index: int = 0,
        ensembles: Iterable[str] = ("nve",),
        temperatures: Iterable[int] = (300,),
        n_steps: Iterable[str] = (10000,),
        file_batch_out: str = "lammps_extrapolations.sh",
        **kwargs,
    ):
        """
        Write batch script for using using LAMMPS to test the number of extrapolations that
        occur during simulation.

        Can also use `**kwargs` to set optional arguments for the SLURM batch script.

        Parameters
        ----------
        n2p2_directory_index: str, optional
            The index of the directory within `self.n2p2_directories` containing the weights
            files. Default is 0.
        ensembles: Iterable[str] = ("nve",)
            Contains the ensembles to run simulations with. Each is applied in turn, and each
            entry should correspond to an entry in `temperatures` and `n_steps` as well.
            Supported strings are "nve", "nvt" and "npt". Default is ("nve",).
        temperatures: Iterable[int] = (300,)
            Contains all temperatures to run simulations at, in Kelvin. Each is applied in
            turn, and each entry should correspond to an entry in `ensembles` and `n_steps`
            as well. Default is (300,).
        n_steps: Iterable[int] = (10000,)
            Contains the number of timesteps to run simulations for. Each is applied in
            turn, and each entry should correspond to an entry in `ensembles` and `tempertures`
            as well. Default is (10000,).
        file_batch_out: str, optional
            File location to write the batch script to relative to
            `scripts_sub_directory`. Default is 'lammps_extrapolations.sh'.
        **kwargs:
            Used to set optional str arguments for the batch script:
              - constraint
              - nodes
              - ntasks_per_node
              - time
              - out
              - account
              - reservation
              - exclusive
              - commands
        """
        file_id = ""
        for i, ensemble in enumerate(ensembles):
            file_id += f"-{ensemble}_t{temperatures[i]}"

        # Create lammps input file
        format_lammps_input(
            formatted_file=f"{self.lammps_directory}/md{file_id}.lmp",
            masses=self.all_structures.mass_map_alphabetical,
            emap=self.all_structures.element_map_alphabetical,
            integrators=ensembles,
            elements=" ".join(self.all_structures.element_list_alphabetical),
            n_steps=n_steps,
            temps=temperatures,
        )

        commands = self.n2p2_module_commands + [
            f"ln -s {self.n2p2_directories[n2p2_directory_index]} {self.lammps_directory}/nnp",
            f"\ncd {self.lammps_directory}",
            (
                "mpirun -np ${SLURM_NTASKS} "
                f"{self.lammps_executable} -in md{file_id}.lmp > md{file_id}.log"
            ),
            "rm nnp",
        ]

        format_slurm_input(
            formatted_file=join_paths(self.scripts_directory, file_batch_out),
            commands=commands,
            job_name="LAMMPS_extrapolations",
            **kwargs,
        )

    def analyse_extrapolations(
        self,
        log_file: str = "md-{ensemble}_t{t}.log",
        ensembles: Iterable[str] = ("nve", "nvt", "npt"),
        temperatures: Iterable[int] = (300,),
        temperature_range: Tuple[int, int] = (0, None),
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
            print("Temp | Timestep | Temperature")
            print("----:|---------:|-----------:")
            for t in temperatures:
                log_file_formatted = log_file.format(ensemble=ensemble, t=t)
                log_file_complete = f"{self.lammps_directory}/{log_file_formatted}"
                timesteps, _, _, lammps_temperatures = read_lammps_log(
                    dump_lammpstrj=1, log_lammps_file=log_file_complete
                )
                temp_mean = np.mean(
                    lammps_temperatures[temperature_range[0] : temperature_range[1]]
                )
                timestep_data[ensemble][t] = timesteps[-1]
                print(f"{t:4d} | {timestep_data[ensemble][t]:8d} | {temp_mean:8.3f}")
            timestep_data[ensemble]["mean"] = np.mean(
                list(timestep_data[ensemble].values())
            )
            print("MEAN | {0:8d}\n".format(int(round(timestep_data[ensemble]["mean"]))))

        return timestep_data

    def reduce_dataset_min_separation(
        self,
        data_file_in: str = "input.data",
        data_file_out: str = "input.data",
        data_file_backup: str = "input.data.minimum_separation_backup",
    ):
        """
        Removes individual frames from `data_file_in` that do not meet the criteria on
        minimum separation set by `structure`. The frames that do satisfy the requirement
        are written to `data_file_out`. To prevent accidental overwrites, the original contents
        of `data_file_in` are also copied to `data_file_backup`.

        Parameters
        ----------
        data_file_in: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directories`,
            to read from. Default is "input.data".
        data_file_out: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directories`,
            to write to. Default is "input.data".
        data_file_backup: str, optional
            File path of the n2p2 structure file, relative to `self.n2p2_directories`, to copy
            the original `data_file_in` to. Default is "input.data.minimum_separation_backup".

        Returns
        -------
        List[int]
            The list of frame indices that have been removed from the Dataset.
        """
        all_remove_indices = []
        for n2p2_directory in self.n2p2_directories:
            dataset = Dataset(
                data_file=join_paths(n2p2_directory, data_file_in),
                all_structures=self.all_structures,
            )

            dataset.write_data_file(
                file_out=join_paths(n2p2_directory, data_file_backup)
            )
            _, removed_indices = dataset.write_data_file(
                file_out=join_paths(n2p2_directory, data_file_out),
                conditions=dataset.check_min_separation_all(),
            )
            print(
                "Removing {} frames for having atoms within minimum separation."
                "".format(len(removed_indices))
            )
            all_remove_indices.append(removed_indices)

        return all_remove_indices

    def remove_n2p2_normalisation(self):
        """
        Removes files associated with the output of nnp-norm, and reverts "input.nn" to
        "input.nn.bak".
        """
        for n2p2_directory in self.n2p2_directories:
            move(
                join_paths(n2p2_directory, "input.nn.bak"),
                join_paths(n2p2_directory, "input.nn"),
            )
            remove(
                join_paths(n2p2_directory, "output.data"),
            )
            remove(
                join_paths(n2p2_directory, "evsv.dat"),
            )
