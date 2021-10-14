from copy import deepcopy
from os import listdir, mkdir
from os.path import isdir, isfile, join
from shutil import copy
from typing import Dict, List, Literal, Tuple, Union
import warnings

import numpy as np

from cc_hdnnp.data import Data
from cc_hdnnp.data_operations import check_nearest_neighbours
from cc_hdnnp.file_operations import read_lammps_log, read_normalisation
from cc_hdnnp.structure import Structure

# TODO combine all unit conversions
# Set a float to define the conversion factor from Bohr radius to Angstrom.
# Recommendation: 0.529177210903 (CODATA 2018).
Bohr2Ang = 0.529177210903
# Set a float to define the conversion factor from Hartree to electronvolt.
# Recommendation: 27.211386245988 (CODATA 2018).
Hartree2eV = 27.211386245988


class ActiveLearning:
    """
    Class for the operations associated with the active learning process.

    Parameters
    ----------
    data_controller : Data
        Used for locations of relevant directories and storing Structure objects.
    integrators : str or list of str, optional
        Set (an array of) string(s) which defines the usage of the "nve", "nvt", and/or "npt"
        integrators. Default is "npt" as this varies the density/simulation cell as well.
    pressures : float or list of float, optional
        Set an array of integers/floats of pressure values in bar used in NpT simulations.
        Set an empty array if no NpT simulations are performed.
        Default is `1.0`. For solids as this normally does not have to be varied.
    N_steps : int, optional
        Set an integer to define the number of MD steps (simulation.lammps has to use the
        variable N_steps as well). Default is `200000`.
    timestep : float, optional
        Set a float to define the timestep in ps.
        Default is `0.0005`, but 0.001 may be more appropriate for systems without H.
    barostat_option : str, optional
        Set a string which specifies the barostat option of the npt integrator: iso, aniso, or
        tri (iso and aniso are not supported in combination with a non orthorhombic cell). If
        the npt integrator is not used, choose the option according to the simulation cell. Non
        orthorhombic cells require to set tri otherwise iso or aniso can be selected (no
        difference if npt integrator is not used). Default is "tri" as this is without any
        restrictions. But dependent on the system there can be good reasons for the usage of
        one of the others.
    atom_style : str, optional
        Set a string which specifies the atom style of LAMMPS structure.lammps file: atomic or
        full. Default is "atomic", as full is currently only required for magnetic HDNNPs
        (requires compiling LAMMPS with the molecule package).
    dump_lammpstrj : int, optional
        Set an integer which defines that only every nth structure is kept in the
        structures.lammpstrj file if no extrapolation occured. The value has to be a divisor of
        N_steps. Default is 200.
    max_len_joblist : int, optional
        Set an integer which specifies the maximal length of the job list including the LAMMPs
        simulations as there might be a limit of the job array size. If the number of jobs is
        higher, several job lists are generated. If the value is set to 0, all jobs are
        compiled in one job list. Default is `0`.
    comment_name_keyword : str, optional
        Set a string which specifies the line start of lines in the input.data file which
        include the name of the structure. This enables to apply different settings for
        different groups of structures, for example, if their training progress is at different
        levels. Furthermore, the name might be required for the assignment of settings in the
        later electronic structure calculations. Avoid '_' in this structure name as this sign
        is used in the following as separator. If a structure name is not required, None has to
        be assigned. If None is assigned, comment_name_index and comment_name_separator will
        not be used. Default is "comment structure".
    runner_cutoff : float, optional
        Set an integer/float to define the RuNNer cutoff radius in Bohr radii.
        Default is `12.0`.
    periodic : bool, optional
        Set to True for periodic systems and to False for non-periodic systems. For
        non-periodic systems an orthorhombic simulation cell with lattice constants (x_max
        - x_min + 2 * runner_cutoff, y_max - y_min + 2 * runner_cutoff, z_max - z_min + 2
        * runner_cutoff) is used in the LAMMPS simulation. Default is `True`.
     tolerances : list of float, optional
        Does not required any modifications for regular usage.
        Set an array of floats which defines the tolerances in increasing order affecting the
        selection of extrapolated structures. The second entry specifies the threshold for the
        sum of the normalised symmetry function value extrapolations of the first selected
        extrapolated structure. If less than 0.1% of the simulations include such an
        extrapolation, the first entry is used instead. The following entries specify tested
        thresholds for the second selected structure. The initially used one is specified by
        initial_tolerance (initial_tolerance = 5 means sixth entry as Python starts counting
        at 0). Its value is increased if the normalised symmetry function value extrapolations
        of first and second selected structures overlap too much or if the minimum time step
        separation criterium is not fulfilled. The tolerance will be decreased if no large
        extrapolations are found or if the structure does not obey the geometrical rules
        specified above. In this way the entries between the second and by initial_tolerance
        specified entry of the array can be selected. The given values yielded good performance
        in all previous tests. If very small extrapolations are a problem, reduce the first two
        values. If there is a large gap between the small and large extrapolations, reduce the
        third to last values. You can also increase the number and density of the third to last
        entry to be more sensitive but with the drawback of a reduced performance. The value of
        initial_tolerance has to be higher than 1. Default is `None`.
    initial_tolerance : int, optional
        Default is `5`.
    """

    def __init__(
        self,
        data_controller: Data,
        integrators: Union[str, List[str]] = "npt",
        pressures: Union[float, List[float]] = 1.0,
        N_steps: int = 200000,
        timestep: float = 0.0005,
        barostat_option: str = "tri",
        atom_style: str = "atomic",
        dump_lammpstrj: int = 200,
        max_len_joblist: int = 0,
        comment_name_keyword: str = "comment structure",
        runner_cutoff: float = 12.0,
        periodic: bool = True,
        tolerances: List[float] = None,
        initial_tolerance: int = 5,
    ):
        if tolerances is None:
            tolerances = [
                0.001,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ]
        if isinstance(integrators, str):
            integrators = [integrators]
        if isinstance(pressures, float):
            pressures = [pressures]

        structures = list(data_controller.all_structures.structure_dict.values())
        self._validate_timesteps(timestep, N_steps, structures)

        if len(data_controller.n2p2_directories) != 2:
            raise ValueError(
                "`data_controller.n2p2_directories` must have 2 entries, but had {}"
                "".format(len(data_controller.n2p2_directories))
            )

        for integrator in integrators:
            if integrator != "nve" and integrator != "nvt" and integrator != "npt":
                raise ValueError(
                    "Integrator {0} is not implemented.".format(integrator)
                )
        self.integrators = integrators

        if not list(pressures) and "npt" in integrators:
            raise ValueError(
                "Integrator npt requires to specify at least one value for pressure."
            )
        self.pressures = pressures

        if (
            barostat_option != "tri"
            and barostat_option != "aniso"
            and barostat_option != "iso"
        ):
            raise ValueError(
                "Barostat option {0} is not implemented.".format(barostat_option)
            )
        self.barostat_option = barostat_option

        if atom_style != "atomic" and atom_style != "full":
            raise ValueError("Atom style {0} is not implemented.".format(atom_style))
        self.atom_style = atom_style

        if N_steps % dump_lammpstrj != 0:
            raise ValueError(
                "N_steps has to be a multiple of dump_lammpstrj ({0}!=N*{1}).".format(
                    N_steps, dump_lammpstrj
                )
            )
        self.N_steps = N_steps

        if any([dump_lammpstrj < s.min_t_separation_interpolation for s in structures]):
            msg = (
                "The extrapolation free structures would be stored only every {0}th time step, "
                "but the minimum time step separation of interpolated structures are set to "
                "{1} time steps.".format(
                    dump_lammpstrj,
                    [s.min_t_separation_interpolation for s in structures],
                )
            )
            raise ValueError(msg)
        self.dump_lammpstrj = dump_lammpstrj

        if not max_len_joblist >= 0:
            msg = (
                "The maximal length of the job list has to be set to 0 (which means infinity) "
                "or a positive integer number, but it was {}".format(max_len_joblist)
            )
            raise ValueError(msg)
        self.max_len_joblist = max_len_joblist

        if (
            comment_name_keyword is None and [s.name for s in structures] is not [None]
        ) or (
            comment_name_keyword is not None and [s.name for s in structures] is [None]
        ):
            msg = (
                "If comment_name_keyword or structure_names is set to None the other one has to"
                " be set to None as well."
            )
            raise ValueError(msg)
        self.comment_name_keyword = comment_name_keyword

        if len(structures) > 1:
            if any(s.name is None for s in structures):
                raise TypeError(
                    "Individual structure names cannot be set to None when using multiple "
                    "Structures."
                )

        if initial_tolerance <= 1:
            raise ValueError("The value of initial_tolerance has to be higher than 1.")

        if len(tolerances) <= initial_tolerance:
            raise ValueError(
                "There are not enough tolerance values as initial_tolerance "
                "results in an index error."
            )

        self.data_controller = data_controller
        self.active_learning_directory = data_controller.active_learning_directory
        self.all_structures = data_controller.all_structures
        self.element_types = self.all_structures.element_list
        self.masses = self.all_structures.mass_list
        self.runner_cutoff = runner_cutoff
        self.periodic = periodic
        self.tolerances = tolerances
        self.initial_tolerance = initial_tolerance

        self.lattices = []
        self.elements = []
        self.charges = []
        self.statistics = []
        self.names = []
        self.positions = []
        self.selection = None

    def _validate_timesteps(
        self, timestep: float, N_steps: int, structures: List[Structure]
    ):
        """
        Given the `timestep` and `N_steps`, validate `min_t_separation_extrapolation`,
        `min_t_separation_interpolation` and `t_separation_interpolation_checks` for each
        `Structure`. If any of these are not set, a default value is determined.

        Parameters
        ----------
        timestep : float
            Define the LAMMPS simulation timestep in ps.
        N_steps : int
            Integer to define the number of MD steps in the LAMMPS simulation.
        structures : list of Structure
            A list of `Structure` objects to check and if needed set the extrapolation and
            interpolation related variables.
        """
        if timestep > 0.01:
            print("WARNING: Very large timestep of {0} ps.".format(timestep))
        self.timestep = timestep
        self.N_steps = N_steps

        for s in structures:
            if s.min_t_separation_extrapolation is None:
                s.min_t_separation_extrapolation = int(0.01 / timestep)

            if s.min_t_separation_interpolation is None:
                s.min_t_separation_interpolation = int(0.1 / timestep)

            if s.t_separation_interpolation_checks is None:
                s.t_separation_interpolation_checks = int(5.0 / timestep)

            if s.t_separation_interpolation_checks * 5 >= N_steps:
                raise ValueError(
                    "`t_separation_interpolation_checks={0}` must be less than a fifth of "
                    "`N_steps={1}` for all structures"
                    "".format(s.t_separation_interpolation_checks, N_steps)
                )
            if s.t_separation_interpolation_checks < s.min_t_separation_interpolation:
                raise ValueError(
                    "`t_separation_interpolation_checks` must be equal to or greater than "
                    "`min_t_separation_interpolation` for all structures"
                )

    def _read_input_data(
        self, comment_name_separator: str = "-", comment_name_index: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads from the n2p2 "input.data" file to determine the names of structures, lattices,
        elements, positions and charges.

        Parameters
        ----------
        comment_name_separator : str, optional
            Can be used to seperate the comment name, which is split by
            `comment_name_seperator`, and the first element taken as the name. Default is "-".
        comment_name_index : int, optional
            Sets the index of the comment name within the line, prior to being split by
            `comment_name_separator`. Default is `2`.
        """
        names = []
        lattices = []
        elements = []
        xyzs = []
        qs = []

        with open(join(self.data_controller.n2p2_directories[0], "input.data")) as f:
            with open(
                join(self.data_controller.n2p2_directories[1], "input.data")
            ) as f_2:
                for line in f.readlines():
                    if line != f_2.readline():
                        raise ValueError(
                            "input.data files in {0} and {1} are differnt.".format(
                                *self.data_controller.n2p2_directories
                            )
                        )
                    line = line.strip()
                    if line.startswith("atom"):
                        line = line.split()
                        elements[-1].append(line[4])
                        xyzs[-1].append([line[1], line[2], line[3]])
                        qs[-1].append(line[5])
                    elif line.startswith("lattice"):
                        lattices[-1].append(line.split()[1:4])
                    elif line.startswith("begin"):
                        lattices.append([])
                        elements.append([])
                        xyzs.append([])
                        qs.append([])
                    elif line.startswith("end"):
                        if not elements[-1]:
                            raise ValueError(
                                "For some of the structures the definition of the atoms is "
                                "incomplete or missing."
                            )
                        xyzs[-1] = np.array(xyzs[-1]).astype(float) * Bohr2Ang
                        qs[-1] = np.array(qs[-1]).astype(float)
                        if self.periodic:
                            if len(lattices[-1]) == 3:
                                lattices[-1] = (
                                    np.array(lattices[-1]).astype(float) * Bohr2Ang
                                )
                            else:
                                raise ValueError(
                                    "The periodic keyword is set to True but for some of the "
                                    "structures the definition of the lattice is incomplete or "
                                    "missing."
                                )
                        else:
                            if lattices[-1]:
                                raise ValueError(
                                    "The periodic keyword is set to False but for some of the "
                                    "structures a definition of a lattice exists."
                                )
                            else:
                                lattices[-1] = np.array(
                                    [
                                        [
                                            xyzs[-1][:, 0].max()
                                            - xyzs[-1][:, 0].min()
                                            + 2 * self.runner_cutoff * Bohr2Ang,
                                            0.0,
                                            0.0,
                                        ],
                                        [
                                            0.0,
                                            xyzs[-1][:, 1].max()
                                            - xyzs[-1][:, 1].min()
                                            + 2 * self.runner_cutoff * Bohr2Ang,
                                            0.0,
                                        ],
                                        [
                                            0.0,
                                            0.0,
                                            xyzs[-1][:, 2].max()
                                            - xyzs[-1][:, 2].min()
                                            + 2 * self.runner_cutoff * Bohr2Ang,
                                        ],
                                    ]
                                )
                    else:
                        if self.comment_name_keyword is not None:
                            if line.startswith(self.comment_name_keyword):
                                names.append(
                                    line.split()[comment_name_index].split(
                                        comment_name_separator
                                    )[0]
                                )

        names = np.array(names)
        lattices = np.array(lattices)
        elements = np.array(elements)
        xyzs = np.array(xyzs)
        qs = np.array(qs)

        return names, lattices, elements, xyzs, qs

    def _write_input_lammps(
        self, path: str, seed: int, temperature: int, pressure: float, integrator: str
    ):
        """
        Prepares an input file for LAMMPS using provided arguments.

        Parameters
        ----------
        path : str
            Path of the folder to save "input.lammps" in.
        seed : int
            Seed for the LAMMPS simulation.
        temperature : int
            Temperature of the simulation in Kelvin.
        pressure : float
            Pressure of the simulation in bar.
        integrator : {"nve", "nvt", "npt"}
            String to set the integrator.
        """
        runner_cutoff = round(self.runner_cutoff * Bohr2Ang, 12)
        cflength = round(1.0 / Bohr2Ang, 12)
        cfenergy = round(1.0 / Hartree2eV, 15)
        elements_string = ""
        for element_type in self.element_types:
            elements_string += element_type + " "

        input_lammps = "variable temperature equal {0}\n".format(float(temperature))
        if integrator == "npt":
            input_lammps += "variable pressure equal {0}\n".format(float(pressure))
        input_lammps += "variable N_steps equal {0}\n".format(
            self.N_steps
        ) + "variable seed equal {0}\n\n".format(seed)
        input_lammps += (
            "units metal\n"
            + "boundary p p p\n"
            + "atom_style {0}\n".format(self.atom_style)
            + "read_data structure.lammps\n"
            + "pair_style nnp dir RuNNer showew yes resetew no maxew 750 showewsum 0 "
            + "cflength {0} cfenergy {1}\n".format(cflength, cfenergy)
            + "pair_coeff * * {0}\n".format(runner_cutoff)
            + "timestep {0}\n".format(self.timestep)
        )
        if integrator == "nve":
            input_lammps += "fix int all nve\n"
        elif integrator == "nvt":
            input_lammps += (
                "fix int all nvt temp ${{temperature}} ${{temperature}} {0}\n".format(
                    self.timestep * 100
                )
            )
        elif integrator == "npt":
            input_lammps += (
                "fix int all npt temp ${{temperature}} ${{temperature}} {0} {1} "
                "${{pressure}} ${{pressure}} {2} fixedpoint 0.0 0.0 0.0\n"
                "".format(
                    self.timestep * 100, self.barostat_option, self.timestep * 1000
                )
            )
        else:
            raise ValueError("Integrator {0} is not implemented.".format(integrator))
        input_lammps += (
            "thermo 1\n"
            + "variable thermo equal 0\n"
            + "thermo_style custom v_thermo step time temp epair etotal fmax fnorm press "
            + "cella cellb cellc cellalpha cellbeta cellgamma density\n"
            + 'thermo_modify format line "thermo '
            + "%8d %10.4f %8.3f %15.5f %15.5f %9.4f %9.4f %9.2f "
            + '%9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %8.5f"\n'
        )
        if self.periodic:
            if self.atom_style == "atomic":
                input_lammps += (
                    "dump lammpstrj all custom 1 structure.lammpstrj id element x y z\n"
                )
            elif self.atom_style == "full":
                input_lammps += (
                    "dump lammpstrj all custom 1 structure.lammpstrj id element "
                    "x y z q\n"
                )
            else:
                raise ValueError(
                    "Atom style {0} is not implemented.".format(self.atom_style)
                )

            input_lammps += (
                "dump_modify lammpstrj pbc yes sort id element {0}\n".format(
                    elements_string[:-1]
                )
            )
        else:
            if self.atom_style == "atomic":
                input_lammps += (
                    "dump lammpstrj all custom 1 structure.lammpstrj id element "
                    "xu yu zu\n"
                )
            elif self.atom_style == "full":
                input_lammps += (
                    "dump lammpstrj all custom 1 structure.lammpstrj id element "
                    "xu yu zu q\n"
                )
            else:
                raise ValueError(
                    "Atom style {0} is not implemented.".format(self.atom_style)
                )

            input_lammps += "dump_modify lammpstrj pbc no sort id element {0}\n".format(
                elements_string[:-1]
            )
        input_lammps += "velocity all create ${temperature} ${seed}\n\n"

        with open(join(self.active_learning_directory, "simulation.lammps")) as f:
            input_lammps += f.read()

        with open(path + "/input.lammps", "w") as f:
            f.write(input_lammps)

    def _write_structure_lammps(
        self,
        path: str,
        lattice: np.ndarray,
        element: np.ndarray,
        xyz: np.ndarray,
        q: np.ndarray,
    ):
        """
        Writes a single structure to file in LAMMPS format from numpy arrays.

        Parameters
        ----------
        path : str
            Directory to write the resultant "structure.lammps" file to.
        lattice : np.ndarray
            Array with the shape (3, 3) representing the dimensions of the structure's lattice.
        element : np.ndarray
            Ordered array of chemical symbols (str) for the atoms in the structure,
            with length equal to the number of atoms present.
        xyz : np.ndarray
            Ordered array of positions with shape (N, 3) for the atoms in the structure,
            with length equal to the number of atoms N.
        q : np.ndarray
            Ordered array of charges for the atoms in the structure, with length equal
            to the number of atoms present.
        """
        lattice_lammps = self.transform_lattice(lattice)

        structure_lammps = (
            "RuNNerActiveLearn\n\n"
            + "{0} atoms\n\n".format(len(element))
            + "{0} atom types\n\n".format(len(self.element_types))
            + "0.0 {0} xlo xhi\n".format(round(lattice_lammps[0], 5))
            + "0.0 {0} ylo yhi\n".format(round(lattice_lammps[1], 5))
            + "0.0 {0} zlo zhi\n".format(round(lattice_lammps[2], 5))
        )
        if self.barostat_option == "tri":
            structure_lammps += "{0} {1} {2} xy xz yz\n".format(
                round(lattice_lammps[3], 5),
                round(lattice_lammps[4], 5),
                round(lattice_lammps[5], 5),
            )
        structure_lammps += "\nMasses\n\n"
        for i in range(len(self.masses)):
            structure_lammps += "{0} {1}\n".format(i + 1, self.masses[i])
        structure_lammps += "\nAtoms\n\n"

        with open(path + "/structure.lammps", "w") as f:
            f.write(structure_lammps)
            if self.atom_style == "atomic":
                for i, element_i in enumerate(element):
                    f.write(
                        "{0:4d} {1} {2:9.5f} {3:9.5f} {4:9.5f}\n".format(
                            i + 1,
                            self.element_types.index(element_i) + 1,
                            xyz[i][0],
                            xyz[i][1],
                            xyz[i][2],
                        )
                    )
            elif self.atom_style == "full":
                for i, element_i in enumerate(element):
                    f.write(
                        "{0:4d} 1 {1} {2:6.3f} {3:9.5f} {4:9.5f} {5:9.5f}\n".format(
                            i + 1,
                            self.element_types.index(element_i) + 1,
                            round(q[i], 3),
                            round(xyz[i][0], 5),
                            round(xyz[i][1], 5),
                            round(xyz[i][2], 5),
                        )
                    )

    def transform_lattice(self, lattice: np.ndarray) -> List[float]:
        """
        Transforms the lattice into a format for LAMMPS. As `lattice` supports arbitrary unit
        vectors, the first lattice vector is taken as lying on the output the x-axis with its
        original magnitude `lx`. `xy` is the projection of the 2nd lattice vector onto the
        output x-axis, and `ly` is the magnitude of the 2nd lattice vector which does not lie
        on the x-axis. Similarly, `xz` and `yz` are projections of the 3rd lattice vector onto
        the x and y axes, with the remaining magnitude expressed as `lz`.

        Parameters
        ----------
        lattice : np.ndarray
            Array with the shape (3, 3).

        Returns
        -------
        list of float
            The transformed components that lie along the cartesian axes, and the projections
            between them.
        """
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])
        cos_alpha = np.dot(lattice[1], lattice[2]) / b / c
        cos_beta = np.dot(lattice[0], lattice[2]) / a / c
        cos_gamma = np.dot(lattice[0], lattice[1]) / a / b
        xy = b * cos_gamma
        lx = a
        xz = c * cos_beta
        ly = np.sqrt(b ** 2 - xy ** 2)
        yz = (b * c * cos_alpha - xy * xz) / ly
        lz = np.sqrt(c ** 2 - xz ** 2 - yz ** 2)

        return [lx, ly, lz, xy, xz, yz]

    def write_lammps(
        self, temperatures: range, seed: int = 1, comment_name_separator: str = "-"
    ):
        """
        Generates the mode1 directory and  LAMMPS files needed to run simulations using the
        two networks.

        Parameters
        ----------
        temperatures : range
            Range of temperature values (in Kelvin) to run simulations at.
        seed : int, optional
            Seed used in the LAMMPS simulations. Is increamented by 1 for each value in
            `self.pressures`.Default is `1`.
        """
        mode1_directory = self.active_learning_directory + "/mode1"
        if isdir(mode1_directory):
            raise IOError(
                "Path mode1 already exists. Please remove old directory first if you would "
                "like to recreate it."
            )
        mkdir(mode1_directory)

        # TODO allow non-default arguments
        (
            names_all,
            lattices_all,
            elements_all,
            xyzs_all,
            qs_all,
        ) = self._read_input_data(comment_name_separator=comment_name_separator)

        if self.max_len_joblist == 0:
            joblist_name = self.active_learning_directory + "/joblist_mode1.dat"
            with open(joblist_name, "w") as f:
                f.write("")

        pressures_npt = self.pressures
        n_simulations = 0
        n_previous_simulations = 0
        counter = 0

        for name, structure in self.all_structures.structure_dict.items():
            if name is None:
                names = names_all
                lattices = lattices_all
                elements = elements_all
                xyzs = xyzs_all
                qs = qs_all
            else:
                names = names_all[names_all == name]
                lattices = lattices_all[names_all == name]
                elements = elements_all[names_all == name]
                xyzs = xyzs_all[names_all == name]
                qs = qs_all[names_all == name]

            structure.selection[0] = structure.selection[0] % structure.selection[1]
            print(
                "Starting from the {0}th structure every {1}th structure of the "
                "input.data file is used.".format(
                    structure.selection[0], structure.selection[1]
                )
            )
            n_structures = len(lattices)
            n_npt = int(np.array([1 for j in self.integrators if j == "npt"]).sum())
            repetitions = max(
                1,
                int(
                    float(n_structures)
                    / 2
                    / len(self.integrators)
                    / len(temperatures)
                    / ((len(self.pressures) - 1) * n_npt / len(self.integrators) + 1)
                    / structure.selection[1]
                ),
            )
            print(
                "The given variations of the settings are repeated {0} times".format(
                    repetitions
                )
            )

            for _ in range(repetitions):
                for j in (0, 1):
                    HDNNP = str(j + 1)
                    nn_file = join(self.data_controller.n2p2_directories[j], "input.nn")
                    scaling_file = join(
                        self.data_controller.n2p2_directories[j], "scaling.data"
                    )
                    if not isfile(nn_file):
                        raise IOError("{} not found".format(nn_file))
                    if not isfile(scaling_file):
                        raise IOError("{} not found".format(scaling_file))

                    for integrator in self.integrators:
                        for temperature in temperatures:
                            if integrator != "npt":
                                pressures = [0]
                            else:
                                pressures = pressures_npt
                            for pressure in pressures:
                                if n_structures // structure.selection[1] <= counter:
                                    n_simulations += counter
                                    counter = 0
                                    print(
                                        "WARNING: The structures of the input.data file are "
                                        "used more than once."
                                    )
                                    if structure.selection[1] > 1:
                                        structure.selection[0] = (
                                            structure.selection[0] + 1
                                        ) % structure.selection[1]
                                        print(
                                            "Try to avoid this by start from the {0}th "
                                            "structure and using again every {1}th structure."
                                            "".format(
                                                structure.selection[0],
                                                structure.selection[1],
                                            )
                                        )
                                selection = (
                                    counter * structure.selection[1]
                                    + structure.selection[0]
                                )
                                path = ""
                                if self.comment_name_keyword is not None:
                                    try:
                                        path += names[selection] + "_"
                                    except IndexError as e:
                                        msg = (
                                            "`names`={0} does not not have entries for the "
                                            "`selection`={1}".format(names, selection)
                                        )
                                        raise IndexError(msg) from e

                                if integrator == "npt":
                                    path += (
                                        integrator
                                        + "_hdnnp"
                                        + HDNNP
                                        + "_t"
                                        + str(temperature)
                                        + "_p"
                                        + str(pressure)
                                        + "_"
                                        + str(seed)
                                    )
                                else:
                                    path += (
                                        integrator
                                        + "_hdnnp"
                                        + HDNNP
                                        + "_t"
                                        + str(temperature)
                                        + "_"
                                        + str(seed)
                                    )

                                mode1_path = mode1_directory + "/" + path
                                if isdir(mode1_path):
                                    raise IOError(
                                        "Path {0} already exists. Please remove old "
                                        "directories first if you would like to recreate them."
                                        "".format(mode1_path)
                                    )
                                mkdir(mode1_path)
                                self._write_input_lammps(
                                    mode1_path, seed, temperature, pressure, integrator
                                )
                                self._write_structure_lammps(
                                    mode1_path,
                                    lattices[selection],
                                    elements[selection],
                                    xyzs[selection],
                                    qs[selection],
                                )
                                mkdir(mode1_path + "/RuNNer")
                                copy(
                                    nn_file,
                                    mode1_path + "/RuNNer/input.nn",
                                )
                                copy(
                                    scaling_file,
                                    mode1_path + "/RuNNer/scaling.data",
                                )
                                atomic_numbers = (
                                    structure.all_species.atomic_number_list
                                )
                                src = join(
                                    self.data_controller.n2p2_directories[j],
                                    "weights.{:03d}.data",
                                )
                                for z in atomic_numbers:
                                    weights_file = src.format(z)
                                    if not isfile(weights_file):
                                        print(
                                            "{} not found, attempting to automatically "
                                            "choose one".format(weights_file)
                                        )
                                        self.data_controller.choose_weights(
                                            n2p2_directory_index=j
                                        )
                                    copy(
                                        weights_file,
                                        mode1_path
                                        + "/RuNNer/weights.{:03d}.data".format(z),
                                    )
                                if (
                                    self.max_len_joblist != 0
                                    and (n_simulations + counter) % self.max_len_joblist
                                    == 0
                                ):
                                    joblist_name = (
                                        self.active_learning_directory
                                        + "/joblist_mode1_"
                                        + str(
                                            (n_simulations + counter)
                                            // self.max_len_joblist
                                            + 1
                                        )
                                        + ".dat"
                                    )
                                    with open(joblist_name, "w") as f:
                                        f.write("")
                                with open(joblist_name, "a") as f:
                                    f.write("{0}\n".format(path))
                                seed += 1
                                counter += 1

            if name is not None:
                n_simulations += counter
                counter = 0
                print(
                    "Input was generated for {0} simulations.".format(
                        n_simulations - n_previous_simulations
                    )
                )
                n_previous_simulations = n_simulations

        if list(self.all_structures.structure_dict.keys())[0] is None:
            n_simulations += counter
            print("Input was generated for {0} simulations.".format(n_simulations))

        self.data_controller._write_active_learning_lammps_script(
            n_simulations=n_simulations
        )

    def _read_lammpstrj(
        self, timesteps: np.ndarray, directory: str
    ) -> Tuple[List[str], int]:
        """
        Reads from "structure.lammpstrj", and returns the lines that correspond to `timesteps`.

        Parameters
        ----------
        timesteps : np.ndarray
            Array of int corresponding to timesteps in the LAMMPS simulation.
        directory : str
            The directory in which to find the "structure.lammpstrj" file.

        Returns
        -------
        list of str, int
            First element is a list of the lines from "structure.lammpstrj" that correspond to
            one of the timesteps in `timesteps`.
            Second element is the number of lines each individual timestep has devoted to it.
        """
        structures = []
        i = 0
        n_timesteps = len(timesteps)
        with open(directory + "/structure.lammpstrj") as f:
            line = f.readline()
            while line and i < n_timesteps:
                # Read structure.lammpstrj until a timestep is found
                while not line.startswith("ITEM: TIMESTEP") and line:
                    line = f.readline()
                line = f.readline()

                # Once a timestep is found and is in `timesteps`, the following lines
                # correspond to the structure at that timestep, until the next timestep
                # line is found
                if timesteps[i] == int(line.strip()):
                    structures.append("ITEM: TIMESTEP\n")
                    while not line.startswith("ITEM: TIMESTEP") and line:
                        structures.append(line)
                        line = f.readline()
                    i += 1

        # Determine how many lines correspond to each timestep/structure
        i = 1
        n_lines = len(structures)
        while i < n_lines and not structures[i].startswith("ITEM: TIMESTEP"):
            i += 1
        structure_lines = i

        return structures, structure_lines

    def _write_lammpstrj(self, structures: List[str], directory: str):
        """
        Writes every line in `structures` to the file "structure.lammpstrj" in `directory`.

        Parameters
        ----------
        structures : list of str
            Each string is written to file as a new line.
        directory : str
            Path of the directory to write in.
        """
        with open(directory + "/structure.lammpstrj", "w") as f:
            for line in structures:
                f.write(line)

    def _write_extrapolation(
        self,
        extrapolation_free_timesteps: int,
        extrapolation_free_lines: int,
        dump_lammpstrj: int,
        structure_lines: int,
        last_timestep: int,
        directory: str,
    ):
        """
        Writes the statistics of extrapolations (provided as arguments) to
        "extrapolation.dat" in `directory`.

        Parameters
        ----------
        extrapolation_free_timesteps : int
            Time steps before the first extrapolation.
        extrapolation_free_lines : int
            Lines (in file) before the first extrapolation.
        dump_lammpstrj : int
            The number of timesteps between dumped, non-extrapolated structures.
        structure_lines : int
            The number of lines per structure.
        last_timestep : int
            The last timestep for the structure.
        directory : str
            Path of the directory to write in.
        """
        with open(directory + "/extrapolation.dat", "w") as f:
            f.write(
                "extrapolation_free_initial_time_steps: {0}\n"
                "lines_before_first_extrapolation: {1}\n"
                "timesteps_between_non_extrapolated_structures: {2}\n"
                "lines_per_structure: {3}\n"
                "last_timestep: {4}".format(
                    extrapolation_free_timesteps,
                    extrapolation_free_lines,
                    dump_lammpstrj,
                    structure_lines,
                    last_timestep,
                )
            )

    def prepare_lammps_trajectory(self):
        """
        Prepares the results of each of the LAMMPS simulations and assesses the number of
        extrapolations.
        """
        for path in listdir(self.active_learning_directory + "/mode1"):
            directory = self.active_learning_directory + "/mode1/" + path
            (
                timesteps,
                extrapolation_free_lines,
                extrapolation_free_timesteps,
            ) = read_lammps_log(
                self.dump_lammpstrj,
                log_lammps_file=join(directory, "log.lammps"),
            )
            structures, structure_lines = self._read_lammpstrj(
                timesteps, directory=directory
            )
            self._write_lammpstrj(structures, directory=directory)
            self._write_extrapolation(
                extrapolation_free_timesteps,
                extrapolation_free_lines,
                self.dump_lammpstrj,
                structure_lines,
                timesteps[-1],
                directory=directory,
            )

    def _get_paths(self, structure_name: str) -> np.ndarray:
        """
        Gets the paths of directories containing finished simulations of `structure_name`.

        Parameters
        ----------
        structure_name : str
            If not empty, only directories matching the strucutre name will be returned.
            Otherwise, all paths are returned.

        Returns
        -------
        np.ndarray
            Array of strings representing the finished simulations' directories.
        """
        paths = []
        if structure_name != "":
            try:
                files = listdir(join(self.active_learning_directory, "mode1"))
                for file in files:
                    if file.startswith(structure_name + "_") and (
                        "_nve_hdnnp" in file
                        or "_nvt_hdnnp" in file
                        or "_npt_hdnnp" in file
                    ):
                        paths.append(file)
            except OSError:
                raise IOError(
                    "Simulations with the structure name {0} were not found.".format(
                        structure_name
                    )
                )
        else:
            files = listdir(join(self.active_learning_directory, "mode1"))
            for file in files:
                if "nve_hdnnp" in file or "nvt_hdnnp" in file or "npt_hdnnp" in file:
                    paths.append(file)

        finished = []
        for i in range(len(paths)):
            if isfile(
                self.active_learning_directory
                + "/mode1/"
                + paths[i]
                + "/extrapolation.dat"
            ):
                finished.append(i)
            else:
                print("Simulation {0} is not finished.".format(paths[i]))
        paths = np.array(paths)[finished]
        if len(paths) == 0:
            raise ValueError(
                "None of the {0} simulations finished.".format(structure_name)
            )

        return paths

    def _read_extrapolation(self, path: str) -> np.ndarray:
        """
        Reads extrapolation statistics for a simulation in the directory given by `path`.

        Parameters
        ----------
        path : str
            The directory within "mode1" which contains the "extrapolation.dat" file.

        Returns
        -------
        np.ndarray
            Array of int. The meaning of each entry is, in order:
              - Time steps before the first extrapolation.
              - Lines (in file) before the first extrapolation.
              - The number of timesteps between dumped, non-extrapolated structures.
              - The number of lines per structure.
              - The last timestep for the structure.
        """
        with open(
            "{0}/mode1/{1}/extrapolation.dat".format(
                self.active_learning_directory, path
            )
        ) as f:
            extrapolation_data = np.array(
                [line.strip().split() for line in f.readlines()]
            )[:, 1].astype(int)

        return extrapolation_data

    def _read_log_format(self, path: str) -> str:
        """
        Determines the format of the "log.lammps" file in mode1 directory `path`.

        Parameters
        ----------
        path : str
            The directory within "mode1" which contains the "log.lammps" file.

        Returns
        -------
        {"v2.0.0", "v2.1.1"}
            str representing the format of the log.
        """
        with open(
            "{0}/mode1/{1}/log.lammps".format(self.active_learning_directory, path)
        ) as f:
            data = [line for line in f.readlines()]
        counter = 0
        n_lines = len(data)
        while counter < n_lines and not data[counter].startswith("**********"):
            counter += 1
        if counter < n_lines:
            if data[counter + 2].startswith("   NNP LIBRARY v2.0.0"):
                extrapolation_format = "v2.0.0"
            elif data[counter + 5].startswith("n²p² version      : v2.1.1") or data[
                counter + 5
            ].startswith("n²p² version  (from git): v2.1.4"):
                extrapolation_format = "v2.1.1"
            else:
                raise IOError(
                    "n2p2 extrapolation warning format cannot be identified in the file "
                    "{0}/log.lammps. Known formats are corresponding to n2p2 v2.0.0 and v2.1.1."
                    "".format(path)
                )

        else:
            raise IOError(
                "n2p2 extrapolation warning format cannot be identified in the file "
                "{0}/log.lammps. Known formats are corresponding to n2p2 v2.0.0 and v2.1.1."
                "".format(path)
            )

        return extrapolation_format

    def _read_log(
        self, path: str, extrapolation_data: np.ndarray, extrapolation_format: str
    ) -> Tuple[List[int], List[float], List[List[List[Union[float, int]]]]]:
        """
        Reads "log.lammps" in `path` and returns information about the extrapolations that
        occured at each tolerance level, identifying the timestep it occured at, the total
        normalised extrapolation and each individual contribtion from an atom/symfunc.

        Parameters
        ----------
        path : str
            The directory within "mode1" which contains the "log.lammps" file.
        extrapolation_data : np.ndarray
            Array of int. The meaning of each entry is, in order:
              - Time steps before the first extrapolation.
              - Lines (in file) before the first extrapolation.
              - The number of timesteps between dumped, non-extrapolated structures.
              - The number of lines per structure.
              - The last timestep for the structure.
        extrapolation_format : {"v2.0.0", "v2.1.1"}
            str representing the format of the log.

        Returns
        -------
        list of int, list of float, list of list of list of float, int, int
            Each list has the length of self.tolerances, and corresponds to extrapolations
            beyond that level. This first list contains the int of a timestep which violated
            the tolerance. The second is the normalised sum of extrapolations that occured
            during the aforementioned timestep. Finally, a list with an entry for each
            tolerance value, which is in turn a list of the individual extrapolations,
            represented as a final list of the normalised extrapolation value (float), the atom
            index (int) and the symfunc index (int) in that order.
        """
        if extrapolation_data[1] != -1:
            # If the line of the first extrapolation is defined, start reading from there.
            # Last line is skipped, as it should correspond to the extrapolation warning error.
            with open(
                "{0}/mode1/{1}/log.lammps".format(self.active_learning_directory, path)
            ) as f:
                data_all = [line.strip() for line in f.readlines()]
                data = data_all[extrapolation_data[1] : -1]

            if extrapolation_format == "v2.0.0":
                data = np.array(
                    [
                        # "thermo" marks a timestep, which is taken as the first element.
                        # the rest are nan
                        [float(line.split()[1])]
                        + [np.nan, np.nan, np.nan, np.nan, np.nan]
                        if line.startswith("thermo")
                        # Otherwise, we have an extrapolation warning. First element is nan,
                        # the rest are:
                        #   - value of symfunc
                        #   - minimum (allowed) symfunc value
                        #   - maximum (allowed) symfunc value
                        #   - atom index
                        #   - symfunc index
                        else [np.nan]
                        + list(
                            np.array(line.split())[[12, 14, 16, 8, 10]].astype(float)
                        )
                        for line in data
                        if line.startswith("thermo") or line.startswith("### NNP")
                    ]
                )
            elif extrapolation_format == "v2.1.1":
                data = np.array(
                    [
                        # "thermo" marks a timestep, which is taken as the first element.
                        # the rest are nan
                        [float(line.split()[1])]
                        + [np.nan, np.nan, np.nan, np.nan, np.nan]
                        if line.startswith("thermo")
                        # Otherwise, we have an extrapolation warning. First element is nan,
                        # the rest are:
                        #   - value of symfunc
                        #   - minimum (allowed) symfunc value
                        #   - maximum (allowed) symfunc value
                        #   - atom index
                        #   - symfunc index
                        else [np.nan]
                        + list(
                            np.array(line.split())[[16, 18, 20, 8, 12]].astype(float)
                        )
                        for line in data
                        if line.startswith("thermo") or line.startswith("### NNP")
                    ]
                )

            if np.isnan(data[-1, 0]):
                # If the last data entry was an extrapolation warning, crop out the last
                # entries that occured, as we did not complete this final timestep.
                data = data[: -np.argmax(np.isfinite(data[:, 0][::-1]))]

            if np.isnan(data[0, 0]):
                # If the first data entry is an extrapolation warning, then add in a dummy
                # timestep entry corresponding to the -1th step to the start of data.
                print(
                    "WARNING: Extrapolation occurred already in the first time step in {0}."
                    "".format(path)
                )
                data = np.concatenate(
                    (
                        np.array([[-1.0] + [np.nan, np.nan, np.nan, np.nan, np.nan]]),
                        data,
                    ),
                    axis=0,
                )

            # extrapolation is the deviation outside the allowed symfunc range, normalised
            # to the range between the max and min allowed values, for each extrapolation
            # in data. It is always positive as values that are <= min or >= max do not
            # generate the warnings.
            extrapolation = (
                np.absolute((data[:, 1] - data[:, 2]) / (data[:, 3] - data[:, 2]) - 0.5)
                - 0.5
            )

            for i in range(1, len(extrapolation)):
                # Sum all the extrapolation that has occured at each timestep. Entries that
                # are nan correspond to the timestep entries in data.
                if np.isfinite(extrapolation[i]) and np.isfinite(extrapolation[i - 1]):
                    extrapolation[i] += extrapolation[i - 1]

            extrapolation_indices = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for tolerance in self.tolerances:
                    # Append an index that extrapolates more than each tolerance level
                    extrapolation_indices.append(np.argmax(extrapolation > tolerance))
            extrapolation_timestep = []
            extrapolation_value = []
            extrapolation_statistic = []
            for i in range(len(self.tolerances)):
                if extrapolation_indices[i] > 0:
                    # If we have a valid index for this tolerance level, calculate further
                    j = 1
                    while np.isnan(data[extrapolation_indices[i] + j, 0]):
                        # Keep counting the number of extrapolations that occured until we
                        # reach a timestep (first entry in a data entry is nan)
                        j += 1
                    # The timestep these extrapolations occured in
                    extrapolation_timestep.append(
                        int(data[extrapolation_indices[i] + j, 0])
                    )
                    # The sum of the normalised extrapolation for this timestep
                    extrapolation_value.append(
                        extrapolation[extrapolation_indices[i] + j - 1]
                    )
                    # The value of the symfunc, the atom index and symfunc index preceding
                    # the timestep are recorded
                    extrapolation_statistic.append([])
                    j -= 1
                    extrapolation_statistic[-1].append(
                        data[extrapolation_indices[i] + j, [1, 4, 5]]
                    )
                    # This is the same process used to calculate the normalised symfunc value
                    # earlier
                    extrapolation_statistic[-1][-1][0] = (
                        data[extrapolation_indices[i] + j, 1]
                        - data[extrapolation_indices[i] + j, 2]
                    ) / (
                        data[extrapolation_indices[i] + j, 3]
                        - data[extrapolation_indices[i] + j, 2]
                    ) - 0.5
                    if extrapolation_statistic[-1][-1][0] < 0:
                        extrapolation_statistic[-1][-1][0] += 0.5
                    else:
                        extrapolation_statistic[-1][-1][0] -= 0.5
                    j -= 1
                    # Continue this process, recording reach extrapolation seperate, until the
                    # preceding timestep is reached
                    while np.isnan(data[extrapolation_indices[i] + j, 0]):
                        extrapolation_statistic[-1].append(
                            data[extrapolation_indices[i] + j, [1, 4, 5]]
                        )
                        extrapolation_statistic[-1][-1][0] = (
                            data[extrapolation_indices[i] + j, 1]
                            - data[extrapolation_indices[i] + j, 2]
                        ) / (
                            data[extrapolation_indices[i] + j, 3]
                            - data[extrapolation_indices[i] + j, 2]
                        ) - 0.5
                        if extrapolation_statistic[-1][-1][0] < 0:
                            extrapolation_statistic[-1][-1][0] += 0.5
                        else:
                            extrapolation_statistic[-1][-1][0] -= 0.5
                        j -= 1
                else:
                    # If no extrapolations were above this tolerance level, append dummy stats
                    extrapolation_timestep.append(-1)
                    extrapolation_value.append(0)
                    extrapolation_statistic.append([None])
        else:
            # If the line of the first extrapolation is not defined, return dummy statistics
            # with the length of `self.tolerances`
            extrapolation_timestep = len(self.tolerances) * [-1]
            extrapolation_value = len(self.tolerances) * [0]
            extrapolation_statistic = len(self.tolerances) * [None]

        return extrapolation_timestep, extrapolation_value, extrapolation_statistic

    def _get_timesteps(
        self,
        extrapolation_timesteps: np.ndarray,
        extrapolation_values: np.ndarray,
        extrapolation_data: List[np.ndarray],
        structure: Structure,
    ) -> Tuple[
        List[List[Union[int, List[int]]]],
        Union[np.ndarray, List[int]],
        np.ndarray,
        Literal[2],
    ]:
        """
        For each simulation performed in "mode1", compare the number of steps elapsed and the
        nature of extrapiolations that occured to determine the timestep of structures which
        should be added to the dataset.

        Parameters
        ----------
        extrapolation_timesteps : np.ndarray
            Array of int, with shape (N, M) where N is the number of simulations for
            `structure`, M is the length of `self.tolerances`. Each entry is a timestep that
            resulted in extrapolations (or -1 if none occured at that tolerance).
        extrapolation_values : np.ndarray
            Array of float, with shape (N, M) where N is the number of simulations for
            `structure`, M is the length of `self.tolerances`. Each entry is the normalised sum
            of extrapolations for the simulation at a particular timestep
            (given by `extrapolation_timesteps`).
        extrapolation_data : List[np.ndarray]
            Each array in the list corresponds to a simulation. Within each array, there are 5
            int entries corresponding to:
              - Time steps before the first extrapolation.
              - Lines (in file) before the first extrapolation.
              - The number of timesteps between dumped, non-extrapolated structures.
              - The number of lines per structure.
              - The last timestep for the structure.
        structure : Structure
            The structure that the other arguments correspond to.

        Returns
        -------
        list of list of (int or list of int), list or np.ndarray of int, np.ndarray, 2
            First object returned is a list with entries for each simulation, which in turn
            contains a list with entries that are either int (the  timestep of a selected
            interpolation) or a list of int (the timesteps of the chosen small, and all large
            extrapolations).
            Second is a list which gives the index in `self.tolerances` that is appropriate for
            each simulation.
            Third is also the index giving the appropriate tolerance, but only for "small"
            extrapolations.
            Finally, the number of possible small tolerances is returned as an int.
        """
        min_fraction = 0.001
        n_tolerances = len(self.tolerances)
        n_small = 2
        # Index defining "small" extrapolations in `self.tolerances`
        small = n_small - 1
        structure_extrapolation = structure.min_t_separation_extrapolation
        structure_interpolation = structure.min_t_separation_interpolation
        structure_checks = structure.t_separation_interpolation_checks

        # Compare how many structures contain "small" extrapolation (outside of the first
        # `structure_extrapolation` timesteps) with `min_fraction`
        extrapoltation_occured = (
            extrapolation_timesteps[:, small] >= structure_extrapolation
        )
        extrapolated_timesteps = extrapolation_timesteps[:, small][
            extrapoltation_occured
        ]
        if len(extrapolated_timesteps) < min_fraction * len(
            extrapolation_timesteps[:, small]
        ):
            print(
                "Only less than 0.1% of the simulations show an extrapolation if a tolerance "
                "of {0} is employed (the initial {1} time steps are neglected). The tolerance "
                "value is reduced to {2}.".format(
                    self.tolerances[small],
                    structure_extrapolation,
                    self.tolerances[small - 1],
                )
            )
            small -= 1

        # Consider "small" extrapolation including the first `structure_extrapolation`
        # timesteps
        extrapoltation_occured = [extrapolation_timesteps[:, small] >= 0]
        extrapolated_timesteps = extrapolation_timesteps[:, small][
            extrapoltation_occured
        ]
        if not extrapolated_timesteps.any():
            # Set tolerance indices to a dummy value for each structure
            print("There are no small extrapolations.")
            tolerance_indices = len(extrapolation_timesteps) * [-1]
        else:
            n_simulations_extrapolation = len(extrapolated_timesteps)
            n_simulations = len(extrapolation_timesteps[:, small])
            print(
                "Small extrapolations are present in {0} of {1} simulations ({2}%).".format(
                    n_simulations_extrapolation,
                    n_simulations,
                    round(100.0 * n_simulations_extrapolation / n_simulations, 2),
                )
            )
            # NB: Changed from the original implementation. If a timestep is -1, then no
            # extrapoltion occured and the values would be 0 anyway, in which case taking the
            # mean and std would be pointless.
            # extrapolation_values_reduced = extrapolation_values[
            #     extrapolation_timesteps[:, small] == -1
            # ]
            extrapolation_values_reduced = extrapolation_values[
                extrapolation_timesteps[:, small] != -1
            ]
            mean_small = np.mean(extrapolation_values_reduced[:, small])
            std_small = np.std(extrapolation_values_reduced[:, small])
            if np.isfinite(mean_small + std_small):
                criterium = max(mean_small + std_small, self.tolerances[small])
            else:
                criterium = self.tolerances[small]
            # TODO REMOVE
            # print(
            #     criterium,
            #     self.tolerances[self.initial_tolerance],
            #     self.initial_tolerance,
            #     n_tolerances,
            # )
            # NB: Changed from the original implementation, an
            while (
                criterium > self.tolerances[self.initial_tolerance]
                # NB: Changed from the original implementation to avoid index error when
                # reaching the final value in tolerances
                # and self.initial_tolerance < n_tolerances
                and self.initial_tolerance < n_tolerances - 1
            ):
                # Increase self.initial_tolerance until it is greater than our criterium to get
                # the "large" tolerance
                self.initial_tolerance += 1
            while (
                not (
                    extrapolation_timesteps[:, self.initial_tolerance][
                        extrapolation_timesteps[:, self.initial_tolerance]
                        >= structure_extrapolation
                    ]
                ).any()
                and self.initial_tolerance > n_small
            ):
                # Decrease self.initial tolerance until we find a tolerance that has
                # extrapolated structures, or we reach our "small" tolerance
                print(
                    "There are no large extrapolations for a tolerance of {0} (the initial {1} "
                    "time steps are neglected). The tolerance value is reduced to {2}."
                    "".format(
                        self.tolerances[self.initial_tolerance],
                        structure_extrapolation,
                        self.tolerances[self.initial_tolerance - 1],
                    )
                )
                self.initial_tolerance -= 1
            if self.initial_tolerance == n_small:
                if not (
                    extrapolation_timesteps[:, self.initial_tolerance][
                        extrapolation_timesteps[:, self.initial_tolerance] >= 0
                    ]
                ).any():
                    print("There are no large extrapolations.")
            # For each structure, subtract the timestep of the first small extrapolation from
            # the timesteps of all other (not small) extrapolations
            extra_steps = (
                extrapolation_timesteps[:, n_small:].T
                - extrapolation_timesteps[:, small]
            ).T
            # Set negative steps to be the structure_extrapolation minimum
            extra_steps[extra_steps < 0] = structure_extrapolation + 1
            # Only include extra_steps where an extrapolation occured
            extra_steps_reduced = extra_steps[extrapolation_timesteps[:, small] != -1]
            # For structures where extra steps occured, tolerance is the "large" value
            tolerance_indices = self.initial_tolerance * np.ones(
                len(extra_steps), dtype=int
            )
            # Tolerance index is -1 for structures that didn't extrapolate
            tolerance_indices[extrapolation_timesteps[:, small] == -1] = -1
            tolerance_indices_reduced = tolerance_indices[
                extrapolation_timesteps[:, small] != -1
            ]
            for i in range(self.initial_tolerance - n_small, n_tolerances - n_small):
                # Iterate over possible "large" tolerances, increasing for those structures
                # with less extra steps than structure_extrapolation
                tolerance_indices_reduced[
                    extra_steps_reduced[:, i] < structure_extrapolation
                ] += 1
            # Set index to -1 for those structures that did not have sufficient extra_steps
            # at any tolerance
            tolerance_indices_reduced[tolerance_indices_reduced >= n_tolerances] = -1
            tolerance_indices[
                extrapolation_timesteps[:, small] != -1
            ] = tolerance_indices_reduced
            tolerance_indices[tolerance_indices >= small] -= n_small

        selected_timesteps = []
        smalls = small * np.ones(len(extrapolation_timesteps), dtype=int)
        min_interpolated_structure_checks = 3
        for i in range(len(extrapolation_timesteps)):
            if (
                extrapolation_timesteps[i][small] < 0
                and extrapolation_data[i][4] != self.N_steps
            ):
                # Print warnings in the case where the simulation ended early, but we did not
                # satisfy the tolerance for the extrapolation
                print(
                    "WARNING: A simulation ended due to too many extrapolations but no one of "
                    "these was larger than the tolerance of {0}. If this message is printed "
                    "several times you should consider to reduce the first and second entry of "
                    "tolerances.".format(self.tolerances[small])
                )
                if (
                    small > 0
                    and extrapolation_timesteps[i][small - 1] >= structure_extrapolation
                ):
                    smalls[i] = small - 1
                    print(
                        "With the reduction of the tolerance to {0} an extrapolated structure "
                        "could be found in this case.".format(
                            self.tolerances[smalls[i]]
                        )
                    )
            if extrapolation_timesteps[i][smalls[i]] >= 0:
                # We have extrapolations over small tolerance
                if (
                    extrapolation_timesteps[i][smalls[i]]
                    > (min_interpolated_structure_checks + 2) * structure_checks
                ):
                    # If the timesteps for extrapolation is large enough, add interpolated
                    # structures that occured before it in addition to extrapolated structures
                    selected_timesteps.append(
                        list(
                            range(
                                2 * structure_checks,
                                extrapolation_timesteps[i][smalls[i]]
                                - structure_checks
                                + 1,
                                structure_checks,
                            )
                        )
                        + [
                            extrapolation_timesteps[i][smalls[i]],
                            deepcopy(extrapolation_timesteps[i][n_small:]),
                        ]
                    )
                else:
                    # Add the minimum number of interpolated checks
                    small_timestep_separation_interpolation_checks = (
                        (
                            extrapolation_timesteps[i][smalls[i]]
                            // (min_interpolated_structure_checks + 2)
                        )
                        // extrapolation_data[i][2]
                    ) * extrapolation_data[i][2]
                    n_interpolation_checks = min_interpolated_structure_checks
                    while (
                        small_timestep_separation_interpolation_checks
                        < structure_interpolation
                        and n_interpolation_checks > 1
                    ):
                        n_interpolation_checks -= 1
                        small_timestep_separation_interpolation_checks = (
                            extrapolation_timesteps[i][smalls[i]]
                            // (n_interpolation_checks + 2)
                            // extrapolation_data[i][2]
                        ) * extrapolation_data[i][2]
                    if (
                        small_timestep_separation_interpolation_checks
                        > structure_interpolation
                    ):
                        selected_timesteps.append(
                            [
                                j * small_timestep_separation_interpolation_checks
                                for j in range(2, n_interpolation_checks + 2)
                            ]
                            + [
                                extrapolation_timesteps[i][smalls[i]],
                                deepcopy(extrapolation_timesteps[i][n_small:]),
                            ]
                        )
                    else:
                        selected_timesteps.append(
                            [
                                extrapolation_timesteps[i][smalls[i]],
                                deepcopy(extrapolation_timesteps[i][n_small:]),
                            ]
                        )
            else:
                # Attempt to add interpolated structures in lieu of extrapolated ones
                if (
                    extrapolation_data[i][4]
                    > (min_interpolated_structure_checks + 2) * structure_checks
                ):
                    selected_timesteps.append(
                        list(
                            range(
                                2 * structure_checks,
                                extrapolation_data[i][4] + 1,
                                structure_checks,
                            )
                        )
                        + [-1, (n_tolerances - n_small) * [-1]]
                    )
                else:
                    small_timestep_separation_interpolation_checks = (
                        (
                            extrapolation_data[i][4]
                            // (min_interpolated_structure_checks + 2)
                        )
                        // extrapolation_data[i][2]
                    ) * extrapolation_data[i][2]
                    n_interpolation_checks = min_interpolated_structure_checks
                    while (
                        small_timestep_separation_interpolation_checks
                        < structure_interpolation
                        and n_interpolation_checks > 1
                    ):
                        n_interpolation_checks -= 1
                        small_timestep_separation_interpolation_checks = (
                            extrapolation_data[i][4]
                            // (n_interpolation_checks + 2)
                            // extrapolation_data[i][2]
                        ) * extrapolation_data[i][2]
                    if (
                        small_timestep_separation_interpolation_checks
                        > structure_interpolation
                    ):
                        selected_timesteps.append(
                            [
                                j * small_timestep_separation_interpolation_checks
                                for j in range(2, n_interpolation_checks + 2)
                            ]
                            + [
                                (extrapolation_data[i][4] // extrapolation_data[i][2])
                                * extrapolation_data[i][2],
                                -1,
                                (n_tolerances - n_small) * [-1],
                            ]
                        )
                        print(
                            "Included the last regularly dumped structure of the simulation "
                            "as it ended due to too many extrapolations."
                        )
                    else:
                        if (
                            extrapolation_data[i][4] // extrapolation_data[i][2]
                        ) * extrapolation_data[i][2] >= structure_extrapolation:
                            selected_timesteps.append(
                                [
                                    (
                                        extrapolation_data[i][4]
                                        // extrapolation_data[i][2]
                                    )
                                    * extrapolation_data[i][2],
                                    -1,
                                    (n_tolerances - n_small) * [-1],
                                ]
                            )
                            print(
                                "Included the last regularly dumped structure of the "
                                "simulation as it ended due to too many extrapolations."
                            )
                        else:
                            selected_timesteps.append(
                                [-1, (n_tolerances - n_small) * [-1]]
                            )

        return selected_timesteps, tolerance_indices, smalls, n_small

    def _get_structure(
        self, data: List[str]
    ) -> Tuple[List[List[float]], np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts relevant properties from the lines of `data` and returns them as arrays.

        Parameters
        ----------
        data : list of str
            The relevant lines read from a LAMMPS trajectory file

        Returns
        -------
        list of list of float, np.ndarray, np.ndarray, np.ndarray
            First object is a list with three entries, each another three entry list
            representing a lattice vector of the structure.
            Second object is an array of strings representing the chemical symbol for each
            atom in the structure.
            Third is an array where the first dimension indexes the atoms in the structure, and
            the second is length 3 representing the position of that atom in the cartesian
            co-ordinates.
            Fourth element is an array of the charge of each atom in the structure, provided
            the data has atom style "full". Otherwise, zeros are returned.
        """
        lat = np.array([data[5].split(), data[6].split(), data[7].split()]).astype(
            float
        )
        # Account for lattice vectors not aligned with the cartesian axes
        if data[4].startswith("ITEM: BOX BOUNDS xy xz yz pp pp pp"):
            lx = (
                lat[0][1]
                - lat[0][0]
                - np.array([0.0, lat[0][2], lat[1][2], lat[0][2] + lat[1][2]]).max()
                + np.array([0.0, lat[0][2], lat[1][2], lat[0][2] + lat[1][2]]).min()
            )
            ly = (
                lat[1][1]
                - lat[1][0]
                - np.array([0.0, lat[2][2]]).max()
                + np.array([0.0, lat[2][2]]).min()
            )
            lz = lat[2][1] - lat[2][0]
            lattice = [[lx, 0.0, 0.0], [lat[0][2], ly, 0.0], [lat[1][2], lat[2][2], lz]]
        else:
            lattice = [
                [(lat[0][1] - lat[0][0]), 0.0, 0.0],
                [0.0, (lat[1][1] - lat[1][0]), 0.0],
                [0.0, 0.0, (lat[2][1] - lat[2][0])],
            ]

        atom_style = "atomic"
        if data[8].startswith("ITEM: ATOMS id element x y z q") or data[8].startswith(
            "ITEM: ATOMS id element xu yu zu q"
        ):
            atom_style = "full"

        data = np.array([line.split() for line in data[9:]])
        element = deepcopy(data[:, 1])
        position = deepcopy(data[:, 2:5]).astype(float)
        if atom_style == "full":
            charge = deepcopy(data[:, 5]).astype(float)
        else:
            charge = np.zeros(len(element))

        return lattice, element, position, charge

    def _check_structure(
        self,
        lattice: List[List[float]],
        element: np.ndarray,
        position: np.ndarray,
        path: str,
        timestep: int,
        structure: Structure,
    ) -> bool:
        """
        Checks the interatomic distances for each element present in the structure to ensure
        that none are within the minimum requried seperation.

        Parameters
        ----------
        lattice : list of list of float
            A list with three entries, each another three entry list representing a lattice
            vector of the structure.
        element : np.ndarray
            Array of strings representing the chemical symbol for each atom in the structure.
        position : np.ndarray
            Array where the first dimension indexes the atoms in the structure, and
            the second is length 3 representing the position of that atom in the cartesian
            co-ordinates.
        path : str
            The path of the directory containing the simulation in question.
        timestep : int
            The timestep of the simulation that the `data` corresponds to.
        structure : Structure
            The `Structure` present in `data`

        Returns
        -------
        bool
            Whether the arguments were accepted as a valid structure.
        """
        for i, element_i in enumerate(self.element_types):
            for element_j in self.element_types[i:]:
                accepted, d = check_nearest_neighbours(
                    lattice,
                    position[element == element_i],
                    position[element == element_j],
                    element_i == element_j,
                    structure.all_species.get_species(element_i).min_separation[
                        element_j
                    ],
                )
                if not accepted:
                    print(
                        "Too small interatomic distance in {0}_s{1}: {2}-{3}: {4} Ang".format(
                            path,
                            timestep,
                            element_i,
                            element_j,
                            d,
                        )
                    )
                    return False

        return True

    def _read_structure(
        self, data: List[str], path: str, timestep: int, structure: Structure
    ) -> bool:
        """
        For the given `data`, extract the relevant properties and then assess its suitability
        to be used as future datapoint.
        If suitable, then the details are added to:
          - `self.names`
          - `self.lattices`
          - `self.elements`
          - `self.positions`
          - `self.charges`

        Parameters
        ----------
        data : list of str
            The relevant lines read from a LAMMPS trajectory file
        path : str
            The path of the directory containing the simulation in question.
        timestep : int
            The timestep of the simulation that the `data` corresponds to.
        structure : Structure
            The `Structure` present in `data`

        Returns
        -------
        bool
            Whether the `data` was accepted as a valid structure.
        """
        lattice, element, position, charge = self._get_structure(data)
        accepted = self._check_structure(
            lattice, element, position, path, timestep, structure
        )
        if accepted:
            self.names.append(path + "_s" + str(timestep))
            self.lattices.append(lattice)
            self.elements.append(element)
            self.positions.append(position)
            self.charges.append(charge)

        return accepted

    def _read_structures(
        self,
        path: str,
        extrapolation_data: np.ndarray,
        selected_timestep: List[int],
        n_small: int,
        small: int,
        tolerance_index: int,
        extrapolation_statistic: List[List[List[Union[float, int]]]],
        element2index: Dict[str, int],
        structure: Structure,
    ):
        """
        For all interpolation checks, and the small and large extrapolations for the
        simulation associated with `path`, call `_read_structure` to assess its suitability.
        If suitable, then the details are added to:
          - `self.names`
          - `self.lattices`
          - `self.elements`
          - `self.positions`
          - `self.charges`

        Parameters
        ----------
        path : str
            The directory for a simulation undertaken during mode1.
        extrapolation_data : np.ndarray
            Array of int. The meaning of each entry is, in order:
              - Time steps before the first extrapolation.
              - Lines (in file) before the first extrapolation.
              - The number of timesteps between dumped, non-extrapolated structures.
              - The number of lines per structure.
              - The last timestep for the structure.
        selected_timestep : list of int
            The timestep(s) which have been identified for the simulation.
        n_small : int
            The number of tolerances in `self.tolerance` classed as small.
        small : int
            The index correspinding to the small tolerance that is used for this simulation.
        tolerance_index : int
            The index corresponding to the large tolerance that is used for this simulation.
        extrapolation_statistic : list of list of float, int
            The outer list contains an entry for each tolerance value, which is in turn a list
            of each extrapolation that occured for the simulation. The elements of this list
            are lists with three entries: the normalised  extrapolation value (float), the atom
            index (int) and the symfunc index (int).
        element2index : dict of str, int
            Allows conversion between relevant elements chemical symbol as a string and the
            index it has in the system.
        structure : Structure
            The `Structure` that was being simulated during mode1.

        Returns
        -------
        int
            The tolerance index corresponding to the chosen large extrapolation. If one is not
            found, then -1 is returned.
        """
        structure_extrapolation = structure.min_t_separation_extrapolation
        with open(
            "{0}/mode1/{1}/structure.lammpstrj".format(
                self.active_learning_directory, path
            )
        ) as f:
            data = [line.strip() for line in f.readlines()]

        n_interpolation_checks = len(selected_timestep) - 2
        if n_interpolation_checks > 0:
            for i in range(n_interpolation_checks):
                # A timestep of -1 would indicate no information, only consider when >= 0
                if selected_timestep[i] >= 0:
                    # Find the starting line for the interpolation. data[3] is the number
                    # of atoms in the simulation, which may be mistaken for the timestep.
                    if selected_timestep[i] != int(data[3]):
                        start = data.index(str(selected_timestep[i])) - 1
                    else:
                        # In cases where the timestep is the same as the number of atoms, use
                        # `extrapolation_data[3]` (the number of lines per structure) to avoid
                        # using the number of atoms as the start point erroneously.
                        start = 1
                        while selected_timestep[i] != int(data[start]):
                            start += extrapolation_data[3]
                        start -= 1
                    end = start + extrapolation_data[3]
                    accepted = self._read_structure(
                        data[start:end], path, selected_timestep[i], structure
                    )
                    if accepted:
                        self.statistics.append([])
                    data = data[end:]

        # After the interpolations, the last two timesteps are "small" and "large"
        # extrapolations. Only consider if the extrapolation occurs after the first
        # `structure_extrapolation` steps
        if selected_timestep[-2] >= structure_extrapolation:
            if selected_timestep[-2] != int(data[3]):
                start = data.index(str(selected_timestep[-2])) - 1
            else:
                start = 1
                while selected_timestep[-2] != int(data[start]):
                    start += extrapolation_data[3]
                start -= 1
            end = start + extrapolation_data[3]
            accepted = self._read_structure(
                data[start:end], path, selected_timestep[-2], structure
            )

            if accepted:
                # The array of atom indices in `extrapolation_statistics` runs from 1 to the
                # total number of atoms in the system. Using `self.elements` and
                # `element2index`, map this to an index which identifies only the species of
                # atom (e.g. all "H" atoms might have index 1)
                extrapolation_statistic[small] = np.array(
                    extrapolation_statistic[small]
                )
                species_indices = []
                for i in range(len(extrapolation_statistic[small])):
                    species_indices.append(
                        element2index[
                            self.elements[-1][
                                extrapolation_statistic[small][i, 1].astype(int) - 1
                            ]
                        ]
                    )
                extrapolation_statistic[small][:, 1] = np.array(species_indices)

                # Sort the statistics, and format `self.statistics` to have the entries
                # "small", the chemical symbols, the symfunc index and the extrapolation value
                extrapolation_statistic[small] = extrapolation_statistic[small][
                    extrapolation_statistic[small][:, 0].argsort()
                ]
                extrapolation_statistic[small] = extrapolation_statistic[small][
                    extrapolation_statistic[small][:, 2].argsort(kind="mergesort")
                ]
                extrapolation_statistic[small] = extrapolation_statistic[small][
                    extrapolation_statistic[small][:, 1].argsort(kind="mergesort")
                ]
                self.statistics.append(
                    [
                        "small",
                        str(
                            list(
                                np.array(self.element_types)[
                                    extrapolation_statistic[small][:, 1].astype(int)
                                ]
                            )
                        )
                        .strip("[]")
                        .replace("'", ""),
                        str(
                            list(extrapolation_statistic[small][:, 2].astype(int) + 1)
                        ).strip("[]"),
                        str(
                            [round(j, 5) for j in extrapolation_statistic[small][:, 0]]
                        ).strip("[]"),
                    ]
                )

        accepted = False
        # Reduce `tolerance_index` until an acceptable level is found
        # Set to -1 if one canot be found
        while not accepted and tolerance_index >= 0:
            # Only consider if the extrapolation occurs after the first
            # `structure_extrapolation` steps
            if selected_timestep[-1][tolerance_index] >= structure_extrapolation:
                # Determine start/end points
                if selected_timestep[-1][tolerance_index] != int(data[3]):
                    start = data.index(str(selected_timestep[-1][tolerance_index])) - 1
                else:
                    start = 1
                    while selected_timestep[-1][tolerance_index] != int(data[start]):
                        start += extrapolation_data[3]
                    start -= 1
                end = start + extrapolation_data[3]
                accepted = self._read_structure(
                    data[start:end],
                    path,
                    selected_timestep[-1][tolerance_index],
                    structure,
                )
            else:
                tolerance_index = -1
            if not accepted:
                tolerance_index -= 1
                if (
                    selected_timestep[-1][tolerance_index] - structure_extrapolation
                    < selected_timestep[-2]
                ):
                    tolerance_index = -1

        if accepted:
            # The array of atom indices in `extrapolation_statistics` runs from 1 to the
            # total number of atoms in the system. Using `self.elements` and
            # `element2index`, map this to an index which identifies only the species of
            # atom (e.g. all "H" atoms might have index 1)
            extrapolation_statistic[tolerance_index + n_small] = np.array(
                extrapolation_statistic[tolerance_index + n_small]
            )
            extrapolation_statistic[tolerance_index + n_small][:, 1] = np.array(
                [
                    element2index[
                        self.elements[-1][
                            extrapolation_statistic[tolerance_index + n_small][
                                i, 1
                            ].astype(int)
                            - 1
                        ]
                    ]
                    for i in range(
                        len(extrapolation_statistic[tolerance_index + n_small])
                    )
                ]
            )

            # Sort the statistics, and format `self.statistics` to have the entries
            # "large", the chemical symbols, the symfunc index and the extrapolation value
            extrapolation_statistic[
                tolerance_index + n_small
            ] = extrapolation_statistic[tolerance_index + n_small][
                extrapolation_statistic[tolerance_index + n_small][:, 0].argsort()
            ]
            extrapolation_statistic[
                tolerance_index + n_small
            ] = extrapolation_statistic[tolerance_index + n_small][
                extrapolation_statistic[tolerance_index + n_small][:, 2].argsort(
                    kind="mergesort"
                )
            ]
            extrapolation_statistic[
                tolerance_index + n_small
            ] = extrapolation_statistic[tolerance_index + n_small][
                extrapolation_statistic[tolerance_index + n_small][:, 1].argsort(
                    kind="mergesort"
                )
            ]
            self.statistics.append(
                [
                    "large",
                    str(
                        list(
                            np.array(self.element_types)[
                                extrapolation_statistic[tolerance_index + n_small][
                                    :, 1
                                ].astype(int)
                            ]
                        )
                    )
                    .strip("[]")
                    .replace("'", ""),
                    str(
                        list(
                            extrapolation_statistic[tolerance_index + n_small][
                                :, 2
                            ].astype(int)
                            + 1
                        )
                    ).strip("[]"),
                    str(
                        [
                            round(j, 5)
                            for j in extrapolation_statistic[tolerance_index + n_small][
                                :, 0
                            ]
                        ]
                    ).strip("[]"),
                ]
            )

        return tolerance_index

    def _write_data(
        self,
        names: np.ndarray,
        lattices: np.ndarray,
        elements: np.ndarray,
        positions: np.ndarray,
        charges: np.ndarray,
        file_name: str,
        mode: str,
    ):
        """
        Writes the information about datapoints passed as arguments to `file_name`, with `mode`
        (e.g. "a+" for appending to existing files).

        Parameters
        ----------
        names : np.ndarray
           Array of str with length N giving the name of each mode1 path corresponding to the
           simulation that gave rise to the data.
        lattices : np.ndarray
            Three dimensional array of float with shape (N, 3, 3) where the second axis gives
            the three lattice vectors, each with three cartesian co-ordinates.
        elements : np.ndarray
            Two dimensional array of str with shape (N, M) where the second axis gives
            the chemical symbol of each atom in a given structure.
        positions : np.ndarray
            Three dimensional array of float with shape (N, M, 3) where the second axis gives
            the position of each atom, each with three cartesian co-ordinates.
        charges : np.ndarray
            Two dimensional array of float with shape (N, M) giving the charges of each atom.
        file_name : str
            The path of the file to write to.
        mode : str
            The mode to open `file_name` in.
        """
        # Make sure we have a trailing newline if appending to file
        text = ""
        if mode == "a+":
            with open(file_name) as f:
                lines = f.readlines()
                if len(lines) > 0 and lines[-1].strip() != "":
                    text = "\n"

        with open(file_name, mode) as f:
            f.write(text)
            for i in range(len(names)):
                f.write("begin\ncomment file {0}\n".format(names[i]))
                if (
                    self.statistics is not None
                    and i < len(self.statistics)
                    and list(self.statistics[i])
                ):
                    f.write(
                        "comment statistics {0}\n"
                        "comment statistics {1}\n"
                        "comment statistics {2}\n"
                        "comment statistics {3}\n".format(
                            self.statistics[i][0],
                            self.statistics[i][1],
                            self.statistics[i][2],
                            self.statistics[i][3],
                        )
                    )
                if self.periodic:
                    f.write(
                        "lattice {0:>9.5f} {1:>9.5f} {2:>9.5f}\n"
                        "lattice {3:>9.5f} {4:>9.5f} {5:>9.5f}\n"
                        "lattice {6:>9.5f} {7:>9.5f} {8:>9.5f}\n".format(
                            round(lattices[i][0][0] / Bohr2Ang, 5),
                            round(lattices[i][0][1] / Bohr2Ang, 5),
                            round(lattices[i][0][2] / Bohr2Ang, 5),
                            round(lattices[i][1][0] / Bohr2Ang, 5),
                            round(lattices[i][1][1] / Bohr2Ang, 5),
                            round(lattices[i][1][2] / Bohr2Ang, 5),
                            round(lattices[i][2][0] / Bohr2Ang, 5),
                            round(lattices[i][2][1] / Bohr2Ang, 5),
                            round(lattices[i][2][2] / Bohr2Ang, 5),
                        )
                    )
                for j in range(len(elements[i])):
                    f.write(
                        "atom {0:>9.5f} {1:>9.5f} {2:>9.5f} {3:2} {4:>6.3f}"
                        " 0.0 0.0 0.0 0.0\n".format(
                            round(positions[i][j][0] / Bohr2Ang, 5),
                            round(positions[i][j][1] / Bohr2Ang, 5),
                            round(positions[i][j][2] / Bohr2Ang, 5),
                            elements[i][j],
                            charges[i][j],
                        )
                    )
                f.write("energy 0.0\ncharge 0.0\nend\n")

    def _print_reliability(
        self,
        extrapolation_timesteps: np.ndarray,
        smalls: np.ndarray,
        tolerance_indices: np.ndarray,
        paths: np.ndarray,
    ):
        """
        Uses the small (`smalls`) and large (`tolerance_indices`) tolerance indices to
        determine the average (median) number of steps before a small and subsequent large
        extrapolation for the simulations in `paths`.

        Parameters
        ----------
        extrapolation_timesteps : np.ndarray
            Two dimensional array of int with shape (N, M) where N is the length of `paths`
            and M is the length of `self.tolerances`, representing the timestep which exceeds
            each tolerance level for each simulation.
        smalls : np.ndarray
            Array of int with length N, where N is the length of `paths`, representing the
            appropriate small tolerance for each simulation.
        tolerance_indices : np.ndarray
            Array of int with length N, where N is the length of `paths`, giving the index
            of the appropriate large tolerance for each simulation.
        paths : np.ndarray
            Array of str with length N, where N is the length of `paths`, giving the
            directories of mode1 simulations.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Transposing gives a shape (M, N), so indexing with `smalls` gives an (N, N)
            # array where the diagonal entries are the small timesteps for each simulation in
            # order
            small_extrapolation_timesteps = np.diagonal(
                extrapolation_timesteps.T[smalls]
            )
            # remove any timesteps of -1, corresponding to the lack of an extrapolation
            extrapolation_timesteps_reduced = small_extrapolation_timesteps[
                small_extrapolation_timesteps != -1
            ]
            paths_reduced = paths[small_extrapolation_timesteps != -1]
            # Extract the median number of timesteps overall and for each network, accounting
            # for nans that may occur due to not having any extrapolations for a network
            median_small = np.median(extrapolation_timesteps_reduced)
            if np.isfinite(median_small):
                median_small = int(round(median_small, 0))
                median_small_1 = np.median(
                    extrapolation_timesteps_reduced[
                        np.flatnonzero(
                            np.core.defchararray.find(paths_reduced, "_hdnnp1_") != -1
                        )
                    ]
                )
                if np.isfinite(median_small_1):
                    median_small_1 = int(round(median_small_1, 0))
                median_small_2 = np.median(
                    extrapolation_timesteps_reduced[
                        np.flatnonzero(
                            np.core.defchararray.find(paths_reduced, "_hdnnp2_") != -1
                        )
                    ]
                )
                if np.isfinite(median_small_2):
                    median_small_2 = int(round(median_small_2, 0))
                print(
                    "The median number of time steps to a small extrapolation is "
                    "{0} (HDNNP_1: {1}, HDNNP_2: {2}).".format(
                        median_small, median_small_1, median_small_2
                    )
                )

            # Calculate the number of steps between a small and large extrapolation
            extra_steps = (
                np.diagonal(extrapolation_timesteps.T[tolerance_indices])[
                    tolerance_indices >= 0
                ]
                - small_extrapolation_timesteps[tolerance_indices >= 0]
            )
            # If we have extra steps, calculate medians with same precautions as before
            if not np.isscalar(extra_steps):
                paths_reduced = paths[tolerance_indices >= 0]
                median_extra_steps = np.median(extra_steps)
                if np.isfinite(median_extra_steps):
                    median_extra_steps = int(round(median_extra_steps, 0))
                    median_extra_steps_1 = np.median(
                        extra_steps[
                            np.flatnonzero(
                                np.core.defchararray.find(paths_reduced, "_hdnnp1_")
                                != -1
                            )
                        ]
                    )
                    if np.isfinite(median_extra_steps_1):
                        median_extra_steps_1 = int(round(median_extra_steps_1, 0))
                    median_extra_steps_2 = np.median(
                        extra_steps[
                            np.flatnonzero(
                                np.core.defchararray.find(paths_reduced, "_hdnnp2_")
                                != -1
                            )
                        ]
                    )
                    if np.isfinite(median_extra_steps_2):
                        median_extra_steps_2 = int(round(median_extra_steps_2, 0))
                    print(
                        "The median number of time steps between the first and second selected "
                        "extrapolated structure is {0} (HDNNP_1: {1}, HDNNP_2: {2}).".format(
                            median_extra_steps,
                            median_extra_steps_1,
                            median_extra_steps_2,
                        )
                    )

    def _read_data(self, file_name: str):
        """
        Reads `file_name` and sets the following based on the contents:
          - `self.names`
          - `self.lattices`
          - `self.elements`
          - `self.positions`
          - `self.charges`

        Parameters
        ----------
        file_name : str
            The file to read. Should be in n2p2 format.
        """
        names = []
        lattices = []
        elements = []
        positions = []
        charges = []
        statistics = []
        with open(file_name) as f:
            for line in f.readlines():
                if line.startswith("atom"):
                    line = line.split()
                    elements[-1].append(line[4])
                    positions[-1].append(line[1:4])
                    charges[-1].append(line[5])
                elif line.startswith("lattice"):
                    lattices[-1].append(line.strip().split()[1:4])
                elif line.startswith("comment file"):
                    names.append(line.strip().split()[2])
                elif line.startswith("comment statistics"):
                    statistics[-1].append(line.strip()[19:])
                elif line.startswith("begin"):
                    lattices.append([])
                    elements.append([])
                    positions.append([])
                    charges.append([])
                    statistics.append([])
                elif line.startswith("end"):
                    lattices[-1] = np.array(lattices[-1]).astype(float) * Bohr2Ang
                    elements[-1] = np.array(elements[-1])
                    positions[-1] = np.array(positions[-1]).astype(float) * Bohr2Ang
                    charges[-1] = np.array(charges[-1]).astype(float)
        self.names = np.array(names)
        self.lattices = np.array(lattices)
        self.elements = np.array(elements)
        self.positions = np.array(positions)
        self.charges = np.array(charges)
        self.statistics = np.array(statistics)

    def _print_performance(self, n_calculations: int):
        """
        Prints the time taken during mode2 for each network.

        Parameters
        ----------
        n_calculations : int
            The number of frames from simulations that the energy/forces were calculated
            for in mode2.
        """
        time = []
        for input_name in ["/mode2/HDNNP_1/mode_2.out", "/mode2/HDNNP_2/mode_2.out"]:
            with open(self.active_learning_directory + input_name) as f:
                file_time = [
                    line.strip().split()
                    for line in f.readlines()
                    if line.startswith("TIMING Training loop finished:")
                ]
                if len(file_time) > 0 and len(file_time[0]) > 1:
                    time.append(float(file_time[0][-2]))
        unit = ["s", "s"]
        for i in range(2):
            if time[i] >= 60.0:
                time[i] /= 60.0
                unit[i] = "min"
                if time[i] >= 60.0:
                    time[i] /= 60.0
                    unit[i] = "h"
        print(
            "\nTime to calculate {0} structures using RuNNer: "
            "HDNNP_1: {1} {2}, HDNNP_2: {3} {4}."
            "\n".format(
                n_calculations, round(time[0], 2), unit[0], round(time[1], 2), unit[1]
            )
        )

    def _read_energies(self, input_name: str) -> np.ndarray:
        """
        Read the energies from the n2p2 output file provided.

        Parameters
        ----------
        input_name : str
            The file path of the file to read energies from. Should be in a recognisable
            format, namely the "trainpoints" file generated by the network training.

        Returns
        -------
        np.ndarray
            Two dimensional array with shape (N, 2) where N is the number of structures the
            network was evaluated for. Along the second dimension the first element is the
            index of the structure, and the second is the energy associated with it.
        """
        with open(input_name) as f:
            # readline "pops" the first line so all indexes should decrease by 1
            line = f.readline().strip()
            if line.startswith("point"):
                energies = np.array(
                    [line.strip().split()[2] for line in f.readlines()]
                ).astype(float)
                energies = np.dstack((np.arange(len(energies)), energies))[0]
            elif line.startswith("Conf."):
                energies = np.array(
                    [np.array(line.strip().split())[[1, 3]] for line in f.readlines()]
                ).astype(float)
                energies = energies[:, 1] / energies[:, 0]
                energies = np.dstack((np.arange(len(energies)), energies))[0]
            elif line.startswith("###"):
                energies = np.array(
                    [line.strip().split()[-1] for line in f.readlines()[11:]]
                ).astype(float)
                energies = np.dstack((np.arange(len(energies)), energies))[0]
            else:
                raise IOError("Unknown RuNNer format")

        return energies

    def _read_forces(self, input_name: str) -> np.ndarray:
        """
        Read the forces from the n2p2 output file provided.

        Parameters
        ----------
        input_name : str
            The file path of the file to read forces from. Should be in a recognisable
            format, namely the "trainforces" file generated by the network training.

        Returns
        -------
        np.ndarray
            Two dimensional array with shape (3N, 2) where N is the number of structures the
            network was evaluated for. Along the second dimension the first element is the
            index of the structure, and the second is a force associated with it in one of the
            cartesian directions. It is ordered so that the forces associated with index `i`
            appear consequetively, in xyz order:
            [... [i-1, fz(i-1)], [i, fx(i)], [i, fy(i)], [i, fz(i)], [i+1, fx(i+1)], ...]
        """
        with open(input_name) as f:
            line = f.readline().strip()
            if line.startswith("point"):
                forces = np.array(
                    [
                        np.array(line.strip().split())[[0, 4]]
                        for line in f
                        if line.strip()
                    ]
                ).astype(float)
                forces[:, 0] -= 1
            elif line.startswith("Conf."):
                forces = np.array(
                    [
                        np.array(line.strip().split())[[0, 5, 6, 7]]
                        for line in f
                        if line.strip()
                    ]
                ).astype(float)
                forces = np.concatenate(
                    (forces[:, [0, 1]], forces[:, [0, 2]], forces[:, [0, 3]])
                )
                forces[:, 0] -= 1
            elif line.startswith("###"):
                forces = []
                for line in f.readlines()[12:]:
                    text = line.strip().split()
                    forces.append([text[0], text[-1]])
                forces = np.array(forces).astype(float)
            else:
                raise IOError("Unknown RuNNer format")

        return forces

    def _reduce_selection(
        self,
        selection: np.ndarray,
        max_interpolated_structures_per_simulation: int,
        t_separation_interpolation_checks: int,
        steps: List[int],
        indices: List[int],
    ) -> np.ndarray:
        """
        Initially, `selection` is based on energy/force discrepancies alone. However, it may
        have multiple interpolated datapoints that do not meet the criteria for
        `max_interpolated_structures_per_simulation` or `t_separation_interpolation_checks`.
        The returned `selection` has these datapoints removed.

        Parameters
        ----------
        selection : np.ndarray
            Array of int corresponding to the indices selected to add to the dataset.
        max_interpolated_structures_per_simulation : int
            The maximum number of allowed interpolated datapoints to be selected for a given
            simulation.
        t_separation_interpolation_checks : int
            The required seperation any interpolations in time.
        steps : List[int]
            The timesteps of all interpolations that had sufficient seperation in energy/force.
        indices : List[int]
            The indices of all interpolations that had sufficient seperation in energy/force.

        Returns
        -------
        np.ndarray
            The `selection` indices with multiple datapoints for a single simulation removed
        """
        steps = np.array(steps)
        steps_difference = steps[1:] - steps[:-1]
        min_separation = steps_difference.min()
        if min_separation < t_separation_interpolation_checks:
            # If there is not enough seperation, discard the 2nd step
            selection = selection[selection != indices[1]]
        else:
            n_steps = len(steps)
            min_timestep_separation_interpolation_checks = (
                n_steps // max_interpolated_structures_per_simulation + 1
            ) * t_separation_interpolation_checks
            j = 1
            while j < n_steps - 1:
                if steps_difference[j] <= min_timestep_separation_interpolation_checks:
                    selection = selection[selection != indices[j]]
                    j += 1
                j += 1

        return selection

    def _improve_selection(
        self,
        selection: np.ndarray,
        statistics: np.ndarray,
        names: np.ndarray,
        ordered_structure_names: np.ndarray,
    ) -> np.ndarray:
        """
        Initially, `selection` is based on energy/force discrepancies alone. Using the
        extrapolation `statistics` this is expanded with appropriate extrapolated datapoints,
        as well as removing redundant datapoints from the `selection`.

        Parameters
        ----------
        selection : np.ndarray
            Array of int corresponding to the indices selected to add to the dataset.
        statistics : np.ndarray
            Two dimensional array of str representing the statistics for extrapolations
            that were selected. Shape is (N, 4) where N is number of extrapolations, and the
            entries in the second dimension are either "large" or "small", the chemical symbol
            of the element(s) in question, the symfunc index, and the extrapolation value.
        names : np.ndarray
            Array of str, the names of the directories from mode1.
        ordered_structure_names : np.ndarray
            Array of str, the names of the structures from mode1. Is in the same order
            as `names`, with the same length.

        Returns
        -------
        np.ndarray
            The `selection` indices with indices added for any appropriate extrapolations
            and multiple datapoints for a single simulation removed according to
            `_reduce_selection`.
        """
        current_name = None
        steps = []
        indices = []
        for i, structure_name in enumerate(ordered_structure_names):
            structure = self.all_structures.structure_dict[structure_name]
            if list(statistics[i]) and structure.all_extrapolated_structures:
                # If for a given simulated structure we have statistics and add
                # `all_extrapolated_structures` for it, append to selection.
                selection = np.append(selection, i)
            elif i in selection:
                # "_s" denotes the timestep in `names`
                name_split = names[i].split("_s")
                if current_name == name_split[0]:
                    # If we are already considering this `current_name`, append
                    steps.append(int(name_split[1]))
                    indices.append(i)
                else:
                    # If we are not yet considering this `current_name`, first reduce for the
                    # previous `current_name` if we had multiple occurances for that name
                    if len(steps) > 2:
                        selection = self._reduce_selection(
                            selection,
                            structure.max_interpolated_structures_per_simulation,
                            structure.t_separation_interpolation_checks,
                            steps,
                            indices,
                        )
                    current_name = name_split[0]
                    steps = [int(name_split[1])]
                    indices = [i]

        # Reduce for the final `current_name` if we had multiple occurances for that name
        if len(steps) > 2:
            selection = self._reduce_selection(
                selection,
                structure.max_interpolated_structures_per_simulation,
                structure.t_separation_interpolation_checks,
                steps,
                indices,
            )
        selection = np.unique(selection)

        max_extrapolated_structures = [
            s.max_extrapolated_structures
            for s in self.all_structures.structure_dict.values()
        ]
        exceptions = [s.exceptions for s in self.all_structures.structure_dict.values()]
        if any(max_extrapolated_structures) or any(exceptions):
            # Reduce names to the updated selection, and those with "small" extrapolation
            statistics_reduced = statistics[selection]
            structure_names_reduced = ordered_structure_names[selection]
            structure_names_reduced = np.array(
                [
                    structure_names_reduced[i]
                    for i in range(len(structure_names_reduced))
                    if list(statistics_reduced[i])
                    and statistics_reduced[i][0] == "small"
                ]
            ).astype(str)

            if list(structure_names_reduced):
                # If we still have names, reduce statistics using same criteria
                statistics_reduced = np.array(
                    [i for i in statistics_reduced if list(i) and i[0] == "small"]
                )
                statistics_reduced = np.core.defchararray.add(
                    np.core.defchararray.add(
                        np.core.defchararray.add(
                            np.core.defchararray.add(structure_names_reduced, ";"),
                            statistics_reduced[:, 1],
                        ),
                        ";",
                    ),
                    statistics_reduced[:, 2],
                )
                statistics_unique = np.unique(statistics_reduced)
                # Count the number occurances of each unique statistic
                counts = {}
                for i in statistics_unique:
                    counts[i] = 0
                for i in statistics_reduced:
                    counts[i] += 1
                exception_list = {}

                # If we have maximum allowed extrapolations set and more extrapolations,
                # register an exception to randomly accept the maximum number allowed
                if any(max_extrapolated_structures):
                    for i in statistics_unique:
                        structure_name_i = i.split(";")[0]
                        structure_i = self.all_structures.structure_dict[
                            structure_name_i
                        ]
                        if structure_i.max_extrapolated_structures != 0:
                            if counts[i] > structure_i.max_extrapolated_structures:
                                exception_list[i] = np.concatenate(
                                    (
                                        np.ones(
                                            structure_i.max_extrapolated_structures,
                                            dtype=int,
                                        ),
                                        np.zeros(
                                            counts[i]
                                            - structure_i.max_extrapolated_structures,
                                            dtype=int,
                                        ),
                                    )
                                )
                                np.random.shuffle(exception_list[i])
                                print(
                                    "The extrapolation ['{0}', '{1}'] occurred {2} times."
                                    "".format(
                                        i.split(";")[1], i.split(";")[2], counts[i]
                                    )
                                )

                if any(exceptions):
                    exceptions_unique = []
                    for structure in self.all_structures.structure_dict.values():
                        if structure.exceptions is not None:
                            for j in range(len(structure.exceptions)):
                                exceptions_unique.append(
                                    [
                                        structure.name
                                        + ";"
                                        + structure.exceptions[j][0]
                                        + ";"
                                        + structure.exceptions[j][1],
                                        structure.exceptions[j][2],
                                    ]
                                )
                    counts_keys = counts.keys()
                    for i in exceptions_unique:
                        if i[0] in counts_keys:
                            keep = int(round(i[1] * counts[i[0]]))
                            exception_list[i[0]] = np.concatenate(
                                (
                                    np.ones(keep, dtype=int),
                                    np.zeros(counts[i[0]] - keep, dtype=int),
                                )
                            )

                # Remove exceptions for the small extrapolations
                exception_list_keys = exception_list.keys()
                if list(exception_list_keys):
                    structure_names_reduced = structure_names_reduced[selection]
                    statistics_reduced = np.array(list(statistics[selection]))
                    for i in range(len(selection)):
                        if (
                            list(statistics_reduced[i])
                            and statistics_reduced[i][0] == "small"
                        ):
                            key = (
                                str(structure_names_reduced[i])
                                + ";"
                                + statistics_reduced[i][1]
                                + ";"
                                + statistics_reduced[i][2]
                            )
                            if key in exception_list_keys:
                                if exception_list[key][-1] == 0:
                                    selection[i] = -1
                                exception_list[key] = np.delete(
                                    exception_list[key], -1, 0
                                )
                    selection = np.unique(selection)
                    if selection[0] == -1:
                        selection = selection[1:]

        return selection

    def _print_statistics(
        self, selection: np.ndarray, statistics: np.ndarray, names: np.ndarray
    ):
        """
        Prints information on how many of the new datapoints arise from small and large
        extrapolations, for each `Structure` in the original dataset.

        Parameters
        ----------
        selection : np.ndarray
            Array of int corresponding to the indices selected to add to the dataset.
        statistics : np.ndarray
            Two dimensional array of str representing the statistics for extrapolations
            that were selected. Shape is (N, 4) where N is number of extrapolations, and the
            entries in the second dimension are either "large" or "small", the chemical symbol
            of the element(s) in question, the symfunc index, and the extrapolation value.
        names : np.ndarray
            Array of str, the names of the directories that the selected datapoints belong to.
        """
        structure_names = self.all_structures.structure_dict.keys()
        if structure_names is not None and len(structure_names) > 1:
            for structure_name in structure_names:
                print("Structure: {0}".format(structure_name))
                n_extrapolations = int(
                    np.array(
                        [
                            1
                            for name in names[selection]
                            if name.split("_")[0] == structure_name
                        ]
                    ).sum()
                )
                print(
                    "{0} missing structures were identified.".format(n_extrapolations)
                )
                # Reduce to just the first element of `statistics`, "small"/"large"
                statistics_reduced = np.array(
                    [
                        statistics[selection][i][0]
                        for i in range(len(statistics[selection]))
                        if names[selection][i].split("_")[0] == structure_name
                        and list(statistics[selection][i])
                    ]
                )
                if len(statistics_reduced) > 0:
                    n_small_extrapolations = int(
                        np.array([1 for i in statistics_reduced if i == "small"]).sum()
                    )
                    n_large_extrapolations = int(
                        np.array([1 for i in statistics_reduced if i == "large"]).sum()
                    )
                    print(
                        "{0} missing structures originate from small extrapolations.\n"
                        "{1} missing structures originate from large extrapolations."
                        "".format(n_small_extrapolations, n_large_extrapolations)
                    )
        else:
            print("{0} missing structures were identified.".format(len(selection)))
            # Reduce to just the first element of `statistics`, "small"/"large"
            statistics_reduced = np.array(
                [i[0] for i in statistics[selection] if list(i)]
            )
            if len(statistics_reduced) > 0:
                n_small_extrapolations = int(
                    np.array([1 for i in statistics_reduced if i == "small"]).sum()
                )
                n_large_extrapolations = int(
                    np.array([1 for i in statistics_reduced if i == "large"]).sum()
                )
                print(
                    "{0} missing structures originate from small extrapolations.\n{1} missing "
                    "structures originate from large extrapolations.".format(
                        n_small_extrapolations, n_large_extrapolations
                    )
                )
        statistics = np.array([i for i in statistics[selection] if list(i)])
        if list(statistics):
            self._analyse_extrapolation_statistics(statistics)

    def _analyse_extrapolation_statistics(self, statistics: np.ndarray):
        """
        For each element present in `statistics`, writes a file
        "extrapolation_statistics_X.dat" in which each line has a symfunc index, followed by
        the value of the extrapolation for that function and element.

        Parameters
        ----------
        statistics: np.ndarray
            Two dimensional array of str representing the statistics for extrapolations
            that were selected. Shape is (N, 4) where N is number of extrapolations, and the
            entries in the second dimension are either "large" or "small", the chemical symbol
            of the element(s) in question, the symfunc index, and the extrapolation value.
        """
        elements = []
        for line in statistics[:, 1]:
            if ", " in line:
                elements.extend(line.split(", "))
            else:
                elements.append(line)
        elements = np.array(elements)

        symmetry_functions = []
        for line in statistics[:, 2]:
            if ", " in line:
                symmetry_functions.extend(line.split(", "))
            else:
                symmetry_functions.append(line)
        symmetry_functions = np.array(symmetry_functions).astype(int)

        values = []
        for line in statistics[:, 3]:
            if ", " in line:
                values.extend(line.split(", "))
            else:
                values.append(line)
        values = np.array(values).astype(float)

        element_list = np.unique(elements)
        for e in element_list:
            symfunc = symmetry_functions[elements == e]
            symfunc_list = np.unique(symfunc)
            with open(
                self.active_learning_directory
                + "/extrapolation_statistics_"
                + e
                + ".dat",
                "w",
            ) as f:
                for s in symfunc_list:
                    val = values[elements == e][symfunc == s]
                    for v in val:
                        f.write("{0} {1}\n".format(s, v))

    def prepare_data_new(self):
        """
        For all structures, determines which timesteps from the mode1 simulations should have
        energy calculations performed (based on the extrapolation and interpolation settings)
        and writes them to "input.data-new". If this file already exists, it is used to set
        variables that would otherwise be calculated from scratch.
        """
        # TODO add in an overwrite check?
        if isfile(join(self.active_learning_directory, "input.data-new")):
            print(
                "input.data-new is already present and data will be read from there.\n"
                "To regenerate input.data-new, delete existing file."
            )
            self._read_data(join(self.active_learning_directory, "input.data-new"))
            return

        self.lattices = []
        self.elements = []
        self.charges = []
        self.statistics = []
        self.names = []
        self.positions = []
        for name, structure in self.all_structures.structure_dict.items():
            # Acquire the extrapolation information from all simulations
            if name is None:
                paths = self._get_paths("")
            else:
                print("Structure: {0}".format(name))
                paths = self._get_paths(name)
            extrapolation_data = []
            extrapolation_timesteps = []
            extrapolation_values = []
            extrapolation_statistics = []
            extrapolation_format = self._read_log_format(paths[0])
            for path in paths:
                extrapolation_data.append(self._read_extrapolation(path))
                (
                    extrapolation_timestep,
                    extrapolation_value,
                    extrapolation_statistic,
                ) = self._read_log(path, extrapolation_data[-1], extrapolation_format)
                extrapolation_timesteps.append(extrapolation_timestep)
                extrapolation_values.append(extrapolation_value)
                extrapolation_statistics.append(extrapolation_statistic)
            extrapolation_timesteps = np.array(extrapolation_timesteps).astype(int)
            extrapolation_values = np.array(extrapolation_values)

            # Use this information to determine which timesteps to use
            (
                selected_timesteps,
                tolerance_indices,
                smalls,
                n_small,
            ) = self._get_timesteps(
                extrapolation_timesteps,
                extrapolation_values,
                extrapolation_data,
                structure,
            )

            # Create dictionary mapping chemical symbols to their index
            element2index = {}
            for j in range(len(self.element_types)):
                element2index[self.element_types[j]] = j

            # Read the actual structures needed from file
            for j in range(len(paths)):
                tolerance_indices[j] = self._read_structures(
                    paths[j],
                    extrapolation_data[j],
                    selected_timesteps[j],
                    n_small,
                    smalls[j],
                    tolerance_indices[j],
                    extrapolation_statistics[j],
                    element2index,
                    structure,
                )
            self._print_reliability(
                extrapolation_timesteps, smalls, tolerance_indices, paths
            )
        self.names = np.array(self.names)
        self.lattices = np.array(self.lattices)
        self.elements = np.array(self.elements)
        self.positions = np.array(self.positions)
        self.charges = np.array(self.charges)
        self.statistics = np.array(self.statistics)
        print("Writing {} names to input.data-new".format(len(self.names)))
        self._write_data(
            self.names,
            self.lattices,
            self.elements,
            self.positions,
            self.charges,
            file_name=self.active_learning_directory + "/input.data-new",
            mode="w",
        )
        self.data_controller._write_active_learning_nn_script(
            n2p2_directories=self.data_controller.n2p2_directories
        )

    def prepare_data_add(self):
        """
        Reads from file the results of running the two networks on the new proposed structures.
        The resultant energy and forces are compared against the thresholds for each structure,
        and the final set of datapoints to add to the training determined and written to file.
        """
        if not isdir(self.active_learning_directory + "/mode2"):
            raise IOError("`mode2` directory not found.")

        # If we do not have the results of `prepare_data_new` in memory, attempt to read from
        # file, and failing that, call `prepare_data_new` directly
        if len(self.names) == 0:
            if isfile(join(self.active_learning_directory, "input.data-new")):
                print("Reading from input.data-new")
                self._read_data(join(self.active_learning_directory, "input.data-new"))
            else:
                print("No data loaded, calling `prepare_data_new`")
                self.prepare_data_new()

        self._print_performance(len(self.names))

        energies_1 = self._read_energies(
            self.active_learning_directory + "/mode2/HDNNP_1/trainpoints.000000.out"
        )
        energies_2 = self._read_energies(
            self.active_learning_directory + "/mode2/HDNNP_2/trainpoints.000000.out"
        )
        forces_1 = self._read_forces(
            self.active_learning_directory + "/mode2/HDNNP_1/trainforces.000000.out"
        )
        forces_2 = self._read_forces(
            self.active_learning_directory + "/mode2/HDNNP_2/trainforces.000000.out"
        )

        # Read normalisation factors from file
        conv_energy_0, conv_length_0 = read_normalisation(
            join(self.data_controller.n2p2_directories[0], "input.nn")
        )
        conv_energy_1, conv_length_1 = read_normalisation(
            join(self.data_controller.n2p2_directories[1], "input.nn")
        )
        if not np.isclose(conv_energy_0, conv_energy_1) or not np.isclose(
            conv_length_0, conv_length_1
        ):
            raise ValueError(
                "Normalisation factors conv_energy={0}, conv_length={1} in {2} are different "
                "to conv_energy={3}, conv_length={4} in {5}".format(
                    conv_energy_0,
                    conv_length_0,
                    self.data_controller.n2p2_directories[0],
                    conv_energy_1,
                    conv_length_1,
                    self.data_controller.n2p2_directories[1],
                )
            )

        dE = []
        dF = []
        ordered_structure_names = []
        for i, name in enumerate(self.names):
            # `name` includes various details seperated by underscores, the first of which is
            # the structure name
            structure_name_i = name.split("_")[0]
            ordered_structure_names.append(structure_name_i)
            for structure_name, structure in self.all_structures.structure_dict.items():
                if structure_name == structure_name_i:
                    dE.append(structure.delta_E)
                    for _ in range(3 * len(self.positions[i])):
                        dF.append(structure.delta_F)
                    break
        # Apply factors to convert from physical to the network's internal units
        dE = np.array(dE) * conv_energy_0
        dF = np.array(dF) * conv_energy_0 / conv_length_0
        ordered_structure_names = np.array(ordered_structure_names)

        if conv_energy_0 != 1.0 or conv_length_0 != 1.0:
            print(
                "`dE` and `dF` converted to internal (normalised) network units: "
                "`dE={0}`, `dF={1}`".format(np.unique(dE), np.unique(dF))
            )

        energies = energies_1[np.absolute(energies_2[:, 1] - energies_1[:, 1]) > dE, 0]
        forces = forces_1[np.absolute(forces_2[:, 1] - forces_1[:, 1]) > dF, 0]
        print(
            "{0} structures identified over energy threshold `dE={1}`:\n{2}".format(
                len(np.unique(energies)), np.unique(dE), np.unique(energies)
            )
        )
        print(
            "{0} structures identified over force threshold `dF={1}`:\n{2}".format(
                len(np.unique(forces)), np.unique(dF), np.unique(forces)
            )
        )
        self.selection = np.unique(np.concatenate((energies, forces)).astype(int))
        self.selection = self._improve_selection(
            self.selection, self.statistics, self.names, ordered_structure_names
        )
        self._write_data(
            self.names[self.selection],
            self.lattices[self.selection],
            self.elements[self.selection],
            self.positions[self.selection],
            self.charges[self.selection],
            file_name=self.active_learning_directory + "/input.data-add",
            mode="w",
        )
        self._print_statistics(self.selection, self.statistics, self.names)
