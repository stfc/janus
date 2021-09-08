from copy import deepcopy
from os import listdir, mkdir
from os.path import isdir, isfile, join
from shutil import copy
from typing import List, Tuple, Union
import warnings

import numpy as np

from cc_hdnnp.data import Data
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
    n2p2_directories : list of str
        Active learning requires two trained networks, which should be located in each of the
        directories provided. Locations are taken relative to `data_controller.main_directory`.
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
        n2p2_directories: List[str],
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

        if len(n2p2_directories) != 2:
            raise ValueError(
                "`n2p2_directories` must have 2 entries, but had {}"
                "".format(len(n2p2_directories))
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
                "Barostat option {0} is not implemented in RuNNerActiveLearn_1.py.".format(
                    barostat_option
                )
            )
        self.barostat_option = barostat_option

        if atom_style != "atomic" and atom_style != "full":
            raise ValueError(
                "Atom style {0} is not implemented in RuNNerActiveLearn_1.py.".format(
                    atom_style
                )
            )
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
                    "Individual structure names cannot be set to None. You have to specify"
                    " an array of structure names or use structure_names = None."
                )

        if initial_tolerance <= 1:
            raise ValueError("The value of initial_tolerance has to be higher than 1.")

        if len(tolerances) <= initial_tolerance:
            raise ValueError(
                "There are not enough tolerance values as initial_tolerance results in an index "
                "error."
            )

        self.data_controller = data_controller
        self.active_learning_directory = data_controller.active_learning_directory
        self.n2p2_directories = n2p2_directories
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
                    "`t_separation_interpolation_checks={0}` must less than a fifth of "
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

        with open(self.n2p2_directories[0] + "/input.data") as f:
            with open(self.n2p2_directories[1] + "/input.data") as f_2:
                for line in f.readlines():
                    if line != f_2.readline():
                        raise ValueError(
                            "input.data files in {0} and {1} are differnt.".format(
                                *self.n2p2_directories
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
            Ordered array of chemical symbols (str) for the atoms in the structure, with length equal
            to the number of atoms present.
        xyz : np.ndarray
            Ordered array of positions with shape (N, 3) for the atoms in the structure, with length equal
            to the number of atoms N.
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

    def write_lammps(self, temperatures: range, seed: int = 1):
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
        ) = self._read_input_data()

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
                    # nn_file = join("../../..", self.n2p2_directories[j], "input.nn")
                    # scaling_file = join("../../..", self.n2p2_directories[j], "scaling.data")
                    nn_file = join(self.n2p2_directories[j], "input.nn")
                    scaling_file = join(self.n2p2_directories[j], "scaling.data")
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
                                        "Path {0} already exists. Please remove old directories "
                                        "first if you would like to recreate them.".format(
                                            mode1_path
                                        )
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
                                # symlink(
                                copy(
                                    nn_file,
                                    mode1_path + "/RuNNer/input.nn",
                                )
                                # symlink(
                                copy(
                                    scaling_file,
                                    mode1_path + "/RuNNer/scaling.data",
                                )
                                atomic_numbers = (
                                    structure.all_species.atomic_number_list
                                )
                                src = join(
                                    # "../../..", self.n2p2_directories[j], "weights.{:03d}.data"
                                    self.n2p2_directories[j],
                                    "weights.{:03d}.data",
                                )
                                for z in atomic_numbers:
                                    weights_file = src.format(z)
                                    if not isfile(weights_file):
                                        print(
                                            "{} not found, attempting to automatically "
                                            "choose one".format(weights_file)
                                        )
                                        self.data_controller.n2p2_directory = (
                                            self.n2p2_directories[j]
                                        )
                                        self.data_controller.choose_weights()
                                    # symlink(
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

    def _read_lammps_log(
        self, dump_lammpstrj: int, directory: str
    ) -> Tuple[np.ndarray, int, int]:
        """
        Reads the "log.lammps" file in `directory` and extracts information about if and at
        what timestep extrapolation of the network potential occured.

        Parameters
        ----------
        dump_lammpstrj : int
            Integer which defines that only every nth structure is kept in the
            structures.lammpstrj file if no extrapolation occured. The value has to be a
            divisor of N_steps.
        directory : str
            The directory in which to find the "log.lammps" file.

        Returns
        -------
        (np.ndarray, int, int)
            First element is array of int corresponding to timesteps, second is the number of
            extrapolation free lines and the third is the timestep that corresponds to that
            line.
        """
        with open(directory + "/log.lammps") as f:
            data = [line for line in f.readlines()]

        if len(data) == 0:
            raise ValueError("{}/log.lammps was empty".format(directory))

        # Count the number of lines that precede the simulation so they can be skipped
        counter = 0
        n_lines = len(data)
        while counter < n_lines and not data[counter].startswith("**********"):
            counter += 1

        # Starting at `counter`, check for extrapolation warnings
        extrapolation = False
        i = counter
        while i < n_lines and not data[i].startswith(
            "### NNP EXTRAPOLATION WARNING ###"
        ):
            i += 1
        if i < n_lines:
            extrapolation = True
        i -= 1

        # The extrapolation warning (or end of simulation) look backwards to see how many steps
        # occured
        while i > counter and not data[i].startswith("thermo"):
            i -= 1
        if extrapolation:
            extrapolation_free_lines = i
            if i > counter:
                extrapolation_free_timesteps = int(data[i].split()[1])
            else:
                extrapolation_free_timesteps = -1
        else:
            extrapolation_free_lines = -1
            extrapolation_free_timesteps = int(data[i].split()[1])

        data = [
            int(line.split()[1]) if line.startswith("thermo") else -1
            for line in data[counter:]
            if line.startswith("thermo")
            or line.startswith("### NNP EXTRAPOLATION WARNING ###")
        ]

        # Subsample using `dump_lammpstrj`
        timesteps = np.unique(
            np.array(
                [
                    data[i]
                    for i in range(1, len(data))
                    if data[i] != -1
                    and (data[i] % dump_lammpstrj == 0 or data[i - 1] == -1)
                ]
            )
        )

        return timesteps, extrapolation_free_lines, extrapolation_free_timesteps

    def _read_lammpstrj(
        self, timesteps: np.ndarray, directory: str
    ) -> Tuple[List[str], int]:
        """

        Parameters
        ----------
        timesteps : np.ndarray
            Array of int corresponding to timesteps in the LAMMPS simulation.
        directory : str
            The directory in which to find the "structure.lammpstrj" file.

        Returns
        -------
        (list of str, int)
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

                if timesteps[i] == int(line.strip()):
                    structures.append("ITEM: TIMESTEP\n")
                    while not line.startswith("ITEM: TIMESTEP") and line:
                        structures.append(line)
                        line = f.readline()
                    i += 1

        i = 1
        n_lines = len(structures)
        while i < n_lines and not structures[i].startswith("ITEM: TIMESTEP"):
            i += 1
        structure_lines = i

        return structures, structure_lines

    def _write_lammpstrj(self, structures, directory):
        """ """
        with open(directory + "/structure.lammpstrj", "w") as f:
            for line in structures:
                f.write(line)

    def _write_extrapolation(
        self,
        extrapolation_free_timesteps,
        extrapolation_free_lines,
        dump_lammpstrj,
        structure_lines,
        last_timestep,
        directory,
    ):
        """ """
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
            ) = self._read_lammps_log(self.dump_lammpstrj, directory=directory)
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

    def _get_paths(self, structure_name):
        """ """
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
                if file.startswith(structure_name + "_") and (
                    "_nve_hdnnp" in file or "_nvt_hdnnp" in file or "_npt_hdnnp" in file
                ):
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

    def _read_extrapolation(self, path):
        """ """
        with open(
            "{0}/mode1/{1}/extrapolation.dat".format(
                self.active_learning_directory, path
            )
        ) as f:
            extrapolation_data = np.array(
                [line.strip().split() for line in f.readlines()]
            )[:, 1].astype(int)

        return extrapolation_data

    def _read_log_format(self, path):
        """ """
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

    def _read_log(self, path, extrapolation_data, extrapolation_format):
        """ """
        if extrapolation_data[1] != -1:
            with open(
                "{0}/mode1/{1}/log.lammps".format(self.active_learning_directory, path)
            ) as f:
                data = [line.strip() for line in f.readlines()][
                    extrapolation_data[1] : -1
                ]

            if extrapolation_format == "v2.0.0":
                data = np.array(
                    [
                        [float(line.split()[1])]
                        + [np.nan, np.nan, np.nan, np.nan, np.nan]
                        if line.startswith("thermo")
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
                        [float(line.split()[1])]
                        + [np.nan, np.nan, np.nan, np.nan, np.nan]
                        if line.startswith("thermo")
                        else [np.nan]
                        + list(
                            np.array(line.split())[[16, 18, 20, 8, 12]].astype(float)
                        )
                        for line in data
                        if line.startswith("thermo") or line.startswith("### NNP")
                    ]
                )

            if np.isnan(data[-1, 0]):
                data = data[: -np.argmax(np.isfinite(data[:, 0][::-1]))]

            if np.isnan(data[0, 0]):
                print(
                    "WARNING: Extrapolation occurred already in the first time step in {0}.".format(
                        path
                    )
                )
                data = np.concatenate(
                    (
                        np.array([[-1.0] + [np.nan, np.nan, np.nan, np.nan, np.nan]]),
                        data,
                    ),
                    axis=0,
                )
            extrapolation = (
                np.absolute((data[:, 1] - data[:, 2]) / (data[:, 3] - data[:, 2]) - 0.5)
                - 0.5
            )

            for i in range(1, len(extrapolation)):
                if np.isfinite(extrapolation[i]) and np.isfinite(extrapolation[i - 1]):
                    extrapolation[i] += extrapolation[i - 1]

            extrapolation_indices = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for tolerance in self.tolerances:
                    extrapolation_indices.append(np.argmax(extrapolation > tolerance))
            extrapolation_timestep = []
            extrapolation_value = []
            extrapolation_statistic = []
            for i in range(len(self.tolerances)):
                if extrapolation_indices[i] > 0:
                    j = 1
                    while np.isnan(data[extrapolation_indices[i] + j, 0]):
                        j += 1
                    extrapolation_timestep.append(
                        int(data[extrapolation_indices[i] + j, 0])
                    )
                    extrapolation_value.append(
                        extrapolation[extrapolation_indices[i] + j - 1]
                    )
                    extrapolation_statistic.append([])
                    j -= 1
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
                    extrapolation_timestep.append(-1)
                    extrapolation_value.append(0)
                    extrapolation_statistic.append([None])
        else:
            extrapolation_timestep = len(self.tolerances) * [-1]
            extrapolation_value = len(self.tolerances) * [0]
            extrapolation_statistic = len(self.tolerances) * [None]

        return extrapolation_timestep, extrapolation_value, extrapolation_statistic

    def _get_timesteps(
        self,
        extrapolation_timesteps,
        extrapolation_values,
        extrapolation_data,
        structure: Structure,
    ):
        """ """
        min_fraction = 0.001
        n_tolerances = len(self.tolerances)
        n_small = 2
        small = n_small - 1
        structure_extrapolation = structure.min_t_separation_extrapolation
        structure_interpolation = structure.min_t_separation_interpolation
        structure_checks = structure.t_separation_interpolation_checks
        if len(
            extrapolation_timesteps[:, small][
                extrapolation_timesteps[:, small] >= structure_extrapolation
            ]
        ) < min_fraction * len(extrapolation_timesteps[:, small]):
            print(
                "Only less than 0.1% of the simulations show an extrapolation if a tolerance of "
                "{0} is employed (the initial {1} time steps are neglected). The tolerance value "
                "is reduced to {2}.".format(
                    self.tolerances[small],
                    structure_extrapolation,
                    self.tolerances[small - 1],
                )
            )
            small -= 1
        if not (
            extrapolation_timesteps[:, small][extrapolation_timesteps[:, small] >= 0]
        ).any():
            print("There are no small extrapolations.")
            tolerance_indices = len(extrapolation_timesteps) * [-1]
        else:
            n_simulations_extrapolation = len(
                extrapolation_timesteps[:, small][
                    extrapolation_timesteps[:, small] >= 0
                ]
            )
            n_simulations = len(extrapolation_timesteps[:, small])
            print(
                "Small extrapolations are present in {0} of {1} simulations ({2}%).".format(
                    n_simulations_extrapolation,
                    n_simulations,
                    round(100.0 * n_simulations_extrapolation / n_simulations, 2),
                )
            )
            extrapolation_values_reduced = extrapolation_values[
                extrapolation_timesteps[:, small] == -1
            ]
            mean_small = np.mean(extrapolation_values_reduced[:, small])
            std_small = np.std(extrapolation_values_reduced[:, small])
            criterium = (
                mean_small
                + std_small
                + max(0, self.tolerances[small] - mean_small + std_small)
            )
            while (
                criterium > self.tolerances[self.initial_tolerance]
                and self.initial_tolerance < n_tolerances
            ):
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
            extra_steps = (
                extrapolation_timesteps[:, n_small:].T
                - extrapolation_timesteps[:, small]
            ).T
            extra_steps[extra_steps < 0] = structure_extrapolation + 1
            extra_steps_reduced = extra_steps[extrapolation_timesteps[:, small] != -1]
            tolerance_indices = self.initial_tolerance * np.ones(
                len(extra_steps), dtype=int
            )
            tolerance_indices[extrapolation_timesteps[:, small] == -1] = -1
            tolerance_indices_reduced = tolerance_indices[
                extrapolation_timesteps[:, small] != -1
            ]
            for i in range(self.initial_tolerance - n_small, n_tolerances - n_small):
                tolerance_indices_reduced[
                    extra_steps_reduced[:, i] < structure_extrapolation
                ] += 1
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
                if (
                    extrapolation_timesteps[i][smalls[i]]
                    > (min_interpolated_structure_checks + 2) * structure_checks
                ):
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
                            "Included the last regularly dumped structure of the simulation as it "
                            "ended due to too many extrapolations."
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
                                "Included the last regularly dumped structure of the simulation as"
                                " it ended due to too many extrapolations."
                            )
                        else:
                            selected_timesteps.append(
                                [-1, (n_tolerances - n_small) * [-1]]
                            )

        return selected_timesteps, tolerance_indices, smalls, n_small

    def _get_structure(self, data):
        """ """
        lat = np.array([data[5].split(), data[6].split(), data[7].split()]).astype(
            float
        )
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

    def _check_nearest_neighbours(self, lat, pos_i, pos_j, ii, d_min):
        """ """
        if len(pos_i) == 0 or len(pos_j) == 0:
            return True, -1

        if pos_i.ndim == 1:
            pos_i = np.array([pos_i])
        if pos_j.ndim == 1:
            pos_j = np.array([pos_j])

        pos = np.array(deepcopy(pos_j))
        pos = np.concatenate(
            (
                pos,
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] - lat[1][0] - lat[2][0],
                        pos[:, 1] - lat[0][1] - lat[1][1] - lat[2][1],
                        pos[:, 2] - lat[0][2] - lat[1][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] - lat[1][0],
                        pos[:, 1] - lat[0][1] - lat[1][1],
                        pos[:, 2] - lat[0][2] - lat[1][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] - lat[1][0] + lat[2][0],
                        pos[:, 1] - lat[0][1] - lat[1][1] + lat[2][1],
                        pos[:, 2] - lat[0][2] - lat[1][2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] - lat[2][0],
                        pos[:, 1] - lat[0][1] - lat[2][1],
                        pos[:, 2] - lat[0][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0],
                        pos[:, 1] - lat[0][1],
                        pos[:, 2] - lat[0][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] + lat[2][0],
                        pos[:, 1] - lat[0][1] + lat[2][1],
                        pos[:, 2] - lat[0][2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] + lat[1][0] - lat[2][0],
                        pos[:, 1] - lat[0][1] + lat[1][1] - lat[2][1],
                        pos[:, 2] - lat[0][2] + lat[1][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] + lat[1][0],
                        pos[:, 1] - lat[0][1] + lat[1][1],
                        pos[:, 2] - lat[0][2] + lat[1][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[0][0] + lat[1][0] + lat[2][0],
                        pos[:, 1] - lat[0][1] + lat[1][1] + lat[2][1],
                        pos[:, 2] - lat[0][2] + lat[1][2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[1][0] - lat[2][0],
                        pos[:, 1] - lat[1][1] - lat[2][1],
                        pos[:, 2] - lat[1][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[1][0],
                        pos[:, 1] - lat[1][1],
                        pos[:, 2] - lat[1][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[1][0] + lat[2][0],
                        pos[:, 1] - lat[1][1] + lat[2][1],
                        pos[:, 2] - lat[1][2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] - lat[2][0],
                        pos[:, 1] - lat[2][1],
                        pos[:, 2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[2][0],
                        pos[:, 1] + lat[2][1],
                        pos[:, 2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[1][0] - lat[2][0],
                        pos[:, 1] + lat[1][1] - lat[2][1],
                        pos[:, 2] + lat[1][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[1][0],
                        pos[:, 1] + lat[1][1],
                        pos[:, 2] + lat[1][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[1][0] + lat[2][0],
                        pos[:, 1] + lat[1][1] + lat[2][1],
                        pos[:, 2] + lat[1][2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] - lat[1][0] - lat[2][0],
                        pos[:, 1] + lat[0][1] - lat[1][1] - lat[2][1],
                        pos[:, 2] + lat[0][2] - lat[1][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] - lat[1][0],
                        pos[:, 1] + lat[0][1] - lat[1][1],
                        pos[:, 2] + lat[0][2] - lat[1][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] - lat[1][0] + lat[2][0],
                        pos[:, 1] + lat[0][1] - lat[1][1] + lat[2][1],
                        pos[:, 2] + lat[0][2] - lat[1][2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] - lat[2][0],
                        pos[:, 1] + lat[0][1] - lat[2][1],
                        pos[:, 2] + lat[0][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0],
                        pos[:, 1] + lat[0][1],
                        pos[:, 2] + lat[0][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] + lat[2][0],
                        pos[:, 1] + lat[0][1] + lat[2][1],
                        pos[:, 2] + lat[0][2] + lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] + lat[1][0] - lat[2][0],
                        pos[:, 1] + lat[0][1] + lat[1][1] - lat[2][1],
                        pos[:, 2] + lat[0][2] + lat[1][2] - lat[2][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] + lat[1][0],
                        pos[:, 1] + lat[0][1] + lat[1][1],
                        pos[:, 2] + lat[0][2] + lat[1][2],
                    )
                )[0],
                np.dstack(
                    (
                        pos[:, 0] + lat[0][0] + lat[1][0] + lat[2][0],
                        pos[:, 1] + lat[0][1] + lat[1][1] + lat[2][1],
                        pos[:, 2] + lat[0][2] + lat[1][2] + lat[2][2],
                    )
                )[0],
            ),
            axis=0,
        )

        # If elements are the same, then the shortest distance will be 0.0 (as the "central"
        # included in the array), so select index 1 instead.
        if ii:
            select = 1
        else:
            select = 0

        for p in pos_i:
            d = np.dstack((pos[:, 0] - p[0], pos[:, 1] - p[1], pos[:, 2] - p[2]))[0]
            d = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
            d = d[d.argsort()[select]]
            if d < d_min:
                return False, d

        return True, -1

    def _check_structure(
        self, lattice, element, position, path, timestep, structure: Structure
    ):
        """ """
        for i, element_i in enumerate(self.element_types):
            for element_j in self.element_types[i:]:
                accepted, d = self._check_nearest_neighbours(
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

    def _read_structure(self, data, path, timestep, structure: Structure):
        """ """
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
        path,
        extrapolation_data,
        selected_timestep,
        n_small,
        small,
        tolerance_index,
        extrapolation_statistic,
        element2index,
        structure: Structure,
    ):
        """ """
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
                if selected_timestep[i] >= 0:
                    if selected_timestep[i] != int(data[3]):
                        start = data.index(str(selected_timestep[i])) - 1
                    else:
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
                extrapolation_statistic[small] = np.array(
                    extrapolation_statistic[small]
                )
                extrapolation_statistic[small][:, 1] = np.array(
                    [
                        element2index[
                            self.elements[-1][
                                extrapolation_statistic[small][i, 1].astype(int) - 1
                            ]
                        ]
                        for i in range(len(extrapolation_statistic[small]))
                    ]
                )
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
        while not accepted and tolerance_index >= 0:
            if selected_timestep[-1][tolerance_index] >= structure_extrapolation:
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
        self, names, lattices, elements, positions, charges, file_name: str, mode: str
    ):
        """ """
        with open(file_name, mode) as f:
            # Make sure we have a trailing newline if appending to file
            if mode == "a+":
                lines = f.readlines()
                if len(lines) > 0 and lines[-1].strip() != "":
                    f.write("\n")

            for i in range(len(names)):
                f.write("begin\ncomment file {0}\n".format(names[i]))
                if list(self.statistics[i]):
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
        self, extrapolation_timesteps, smalls, tolerance_indices, paths
    ):
        """ """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            small_extrapolation_timesteps = np.diagonal(
                extrapolation_timesteps.T[smalls]
            )
            extrapolation_timesteps_reduced = small_extrapolation_timesteps[
                small_extrapolation_timesteps != -1
            ]
            paths_reduced = paths[small_extrapolation_timesteps != -1]
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
            extra_steps = (
                np.diagonal(extrapolation_timesteps.T[tolerance_indices])[
                    tolerance_indices >= 0
                ]
                - small_extrapolation_timesteps[tolerance_indices >= 0]
            )
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
        """ """
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

    def _print_performance(self, n_calculations):
        """ """
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

    def _read_energies(self, input_name):
        """ """
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
        """ """
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

    def _read_normalisation(self, input_name):
        """ """
        with open(input_name) as f:
            lines = f.readlines()[5:7]

        if (
            lines[0].split()[0] == "conv_energy"
            and lines[1].split()[0] == "conv_length"
        ):
            return float(lines[0].split()[1]), float(lines[1].split()[1])

        return 1.0, 1.0

    def _reduce_selection(
        self,
        selection: np.ndarray,
        max_interpolated_structures_per_simulation: int,
        t_separation_interpolation_checks: int,
        steps: List[int],
        indices: List[int],
    ) -> np.ndarray:
        """ """
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
        Parameters
        ----------
        selection : np.ndarray
        statistics : np.ndarray
        names : np.ndarray
        """
        current_name = None
        steps = []
        indices = []
        for i, structure_name in enumerate(ordered_structure_names):
            structure = self.all_structures.structure_dict[structure_name]
            if list(statistics[i]):
                if structure.all_extrapolated_structures:
                    selection = np.append(selection, i)
            elif i in selection:
                name_split = names[i].split("_s")
                if current_name == name_split[0]:
                    steps.append(int(name_split[1]))
                    indices.append(i)
                else:
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
        """ """
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
        """ """
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
        """ """
        # TODO add in an overwrite check?
        if isfile(join(self.active_learning_directory, "input.data-new")):
            print("Reading from input.data-new")
            self._read_data(join(self.active_learning_directory, "input.data-new"))

        self.lattices = []
        self.elements = []
        self.charges = []
        self.statistics = []
        self.names = []
        self.positions = []
        for name, structure in self.all_structures.structure_dict.items():
            if structure.name is None:
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
            element2index = {}
            for j in range(len(self.element_types)):
                element2index[self.element_types[j]] = j
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
            n2p2_directories=self.n2p2_directories
        )

    def prepare_data_add(self):
        """ """
        if not isdir(self.active_learning_directory + "/mode2"):
            raise IOError("`mode2` directory not found.")

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
        conv_energy_0, conv_length_0 = self._read_normalisation(
            join(self.n2p2_directories[0], "input.nn")
        )
        conv_energy_1, conv_length_1 = self._read_normalisation(
            join(self.n2p2_directories[1], "input.nn")
        )
        if conv_energy_0 != conv_energy_1 or conv_length_0 != conv_length_1:
            raise ValueError(
                "Normalisation factors conv_energy={0}, conv_length={1} in {2} are different "
                "to conv_energy={3}, conv_length={4} in {5}".format(
                    conv_energy_0,
                    conv_length_0,
                    self.n2p2_directories[0],
                    conv_energy_1,
                    conv_length_1,
                    self.n2p2_directories[1],
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
                "`dE={0}`, `dF={1}`".format(dE, dF)
            )

        energies = energies_1[np.absolute(energies_2[:, 1] - energies_1[:, 1]) > dE, 0]
        forces = forces_1[np.absolute(forces_2[:, 1] - forces_1[:, 1]) > dF, 0]
        print(
            "{0} structures identified over energy threshold `dE={1}`".format(
                len(np.unique(energies)), dE
            )
        )
        print(
            "{0} structures identified over force threshold `dF={1}`".format(
                len(np.unique(forces)), dF
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
