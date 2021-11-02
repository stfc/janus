"""
Class for containing information about atomic species in the structure of interest.
"""


from copy import deepcopy
from typing import Dict, Iterator, List, Tuple

from ase.atoms import Atoms
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
import numpy as np


class Species:
    """
    Holds information about a single atomic species. One of either `symbol` or `atomic_number`
    must be provided. `mass` will be taken from ASE, unless it is provided in which case that
    take precedence.

    Parameters
    ----------
    symbol : str, optional
        Chemical symbol
    atomic_number : int, optional
        Atomic number
    mass : float, optional
        Mass of the species in AMU.
    valence : int, optional
        The number of valence electrons the species has. This is used when calculating relative
        charges using Quantum Espresso, and so is not needed if using CP2K or if charges are not
        required. Default is None.
    min_separation : dict of [str, float], optional
        The minimum allowed distance between atoms of this species and those of other species.
        The keys should be the chemical symbols of atomic species (including this one), with the
        values being separation in Ang. Default is `None`.
    """

    def __init__(
        self,
        symbol: str = None,
        atomic_number: int = None,
        mass: float = None,
        valence: int = None,
        min_separation: Dict[str, float] = None,
    ):
        if symbol is not None and atomic_number is not None:
            if symbol != chemical_symbols[atomic_number]:
                raise ValueError(
                    "Provided symbol {} does not match provided atomic number {}"
                    "".format(symbol, atomic_number)
                )
            self.symbol = symbol
            self.atomic_number = atomic_number
        elif symbol is not None:
            self.symbol = symbol
            self.atomic_number = atomic_numbers[symbol]
        elif atomic_number is not None:
            self.symbol = chemical_symbols[atomic_number]
            self.atomic_number = atomic_number
        else:
            raise ValueError(
                "At least one of `symbol` or `atomic_number` must be provided.`"
            )

        if mass is not None:
            self.mass = mass
        else:
            self.mass = atomic_masses[self.atomic_number]

        if valence is not None:
            if valence < 0:
                raise ValueError(
                    "`valence` must not be negative, but was {}.".format(valence)
                )
            elif valence > self.atomic_number:
                raise ValueError(
                    "`valence` cannot be greater than the `atomic_number`, but they were "
                    "{}, {}.".format(valence, self.atomic_number)
                )
        self.valence = valence

        self.min_separation = min_separation


class Structure:
    """
    Holds information about a given structure in the reference data, primarily for when active
    learning is performed.

    Parameters
    ----------
    name : str
        Arbitrary name given to the `Structure`, used to identify it in relevant filenames.
    all_species : List[Species]
        The Species that constitute this Structure.
    delta_E : float
        The tolerance for discrepancies in energy when active learning.
        If two networks have an energy difference larger than this for a given structure,
        it will be added to the training set. Units are Hartree per atom.
        Recommended value is either the highest energy error of the training data set
        (excluding outliers) or five times the energy RMSE of the training data set for the
        given structures.
    delta_F : float
        The tolerance for discrepancies in forces when active learning. If two networks have an
        force difference larger than this for a given structure, it will be added to the
        training set. Units are Hartree/Bohr. Recommended value is either 100 times `delta_E`,
        the highest force error of the training data set (excluding outliers) or five times the
        force RMSE of the training data set for the given structures.
    global_separation : float, optional
        If the `min_separation` is not set for all pairs of Species in `all_species`,
        then `global_separation` will be used as a default. Units are Ang. Default is 0.8.
    selection : list of int, optional
        A list where the first entry is the how many structures of this type to skip, and the
        second entry is a sampling rate. For example, [10, 5] would sample every fifth
        structure, ignoring the first 10 and starting with the 11th (index 10). In practice
        this depends on the reliability of the HDNNP. The usage the current input.data file as
        source of new structures is recommended. You should not add more than about a third of
        the current number of structures, i.e., performing too many simulations is not
        efficient. The more often you retrain your HDNNP, the less likely it is too include
        structures fixing the same problem. However, if you include too few structures in every
        iteration, the procedure will also be slow. If you try to find the last gaps in the
        sampled configuration space, you can do more simulations and also longer simulations.
        Default is `None`, in which case every structure is used.
    min_t_separation_extrapolation : int, optional
        Defines the minimal number of time steps between two structures for extrapolation checks
        for this structure.
        Default is `None`, in which case it will be chosen so that a check occurs every 0.01 ps
        of the simulation.
    min_t_separation_interpolation : int, optional
        Defines the minimal number of time steps between two structures for interpolation checks
        for this structure (this can lead to less than three checks).
        Default is `None`, in which case it will be chosen so that a check occurs every 0.1 ps
        of the simulation.
    t_separation_interpolation_checks : int, optional
        Defines the usual time step separations between two structures for interpolation checks
        for this structure (this can be smaller in case only less than three checks would be
        possible). The value has to be smaller than a fifth of the number of steps used in
        active learning. Default is `None`, in which case it will be chosen so that a check
        occurs every 5 ps of simulation.
    all_extrapolated_structures : bool, optional
        Specifies for the structure if all extrapolated structures shall be selected. Otherwise
        they are only selected if they above the energy and force thresholds.
        `max_extrapolated_structures` and `exceptions` can deselect the extrapolated structures.
        Should be set to True if the HDNNP has reached some reliability. Default is `True`.
    max_extrapolated_structures : int, optional
        The maximal number of the same kind of extrapolations for this structure. Set to 0 if no
        limit should be applied. In the initial generations of improvements it can happen that a
        particular symmetry function leads to most of the extrapolations. In this case it makes
        sense to select only a fraction of these structures since they fix all the same problem.
        A value about 50 might be reasonable. Default is `50`.
    max_interpolated_structures_per_simulation : int
       Defines the usual maximal numbers of selected interpolated structures per simulation for
       this structure. Default is `4`.
    exceptions : List[list]
        Manually define exceptions of small extrapolations which shall be only included to a
        certain fraction for each structure name. The max_extrapolated_structures limitation is
        overwritten for the given extrapolations. An example for the format is [[A, B, C], ...]
        where A is a string of the element symbols specifying the central atoms of the
        extrapolated symmetry functions, B is a string of the numbers of corresponding symmetry
        functions, and C is a float of the used fraction. A and B have to be identical to the
        entries in input.data-new.
        For each structure an array of several exceptions ([A, B, C]) can be given or None has
        to be set. Candidates can be found using the information given in input.data-new and
        extrapolation_statistics_XX.dat. Default is `None`,
    """

    def __init__(
        self,
        name: str,
        all_species: List[Species],
        delta_E: float,
        delta_F: float,
        global_separation: float = 0.8,
        selection: List[int] = None,
        min_t_separation_extrapolation: int = None,
        min_t_separation_interpolation: int = None,
        t_separation_interpolation_checks: int = None,
        all_extrapolated_structures: bool = True,
        max_extrapolated_structures: int = 50,
        max_interpolated_structures_per_simulation: int = 4,
        exceptions: List[list] = None,
    ):
        self.name = name
        self.all_species = all_species
        self.all_species.sort(key=lambda x: x.atomic_number)
        self._validate_min_separation(global_separation=global_separation)
        self.delta_E = delta_E
        self.delta_F = delta_F

        if selection is None:
            self.selection = [0, 1]
        else:
            if len(selection) != 2:
                raise ValueError(
                    "`selection` must have 2 entries, but was {}".format(selection)
                )
            if not (isinstance(selection[0], int) and isinstance(selection[1], int)):
                raise TypeError(
                    "`selection` entries must have type `int`, but were of type "
                    "{0} and {1}".format(type(selection[0]), type(selection[1]))
                )
            if selection[0] < 0 and selection[1] < 1:
                raise ValueError(
                    "`selection` entries must be at least 0 and 1 respectively, but were "
                    "{0} and {1}".format(selection[0], selection[1])
                )
            self.selection = selection

        # Leave timestep separation variables as None, as a sensible default value will depend
        # on the total length of the simulation being run.
        self.min_t_separation_extrapolation = min_t_separation_extrapolation
        self.min_t_separation_interpolation = min_t_separation_interpolation
        self.t_separation_interpolation_checks = t_separation_interpolation_checks

        self.all_extrapolated_structures = all_extrapolated_structures
        self.max_extrapolated_structures = max_extrapolated_structures
        if not isinstance(max_interpolated_structures_per_simulation, int):
            raise TypeError(
                "`max_interpolated_structures_per_simulation` must have type `int`"
            )
        if max_interpolated_structures_per_simulation < 0:
            raise ValueError(
                "`max_interpolated_structures_per_simulation` must have a value of at least 0"
            )
        self.max_interpolated_structures_per_simulation = (
            max_interpolated_structures_per_simulation
        )
        self.exceptions = exceptions

    @property
    def element_list(self):
        """
        Get the symbols of all species, sorted by atomic number.

        Returns
        -------
        list of str
        """
        return [single_species.symbol for single_species in self.all_species]

    @property
    def atomic_number_list(self) -> List[int]:
        """
        Get the atomic numbers of all species, in order.

        Returns
        -------
        list of int
        """
        return [single_species.atomic_number for single_species in self.all_species]

    @property
    def mass_list(self):
        """
        Get the masses of all species, sorted by atomic number.

        Returns
        -------
        list of str
        """
        return [single_species.mass for single_species in self.all_species]

    def _validate_min_separation(self, global_separation: float):
        """
        Ensures that the `min_separation` is set for all possible combinations of Species.
        Values already set are preserved, and `global_separation` is used otherwise.

        Parameters
        ----------
        global_separation: float
            If the `min_separation` is not set for all pairs of Species in `all_species`,
            then `global_separation` will be used as a default. Units are Ang.
        """
        for single_species in self.all_species:
            if single_species.min_separation is None:
                # If `min_separation` is not set, use the global value for all species
                single_species.min_separation = {
                    symbol: global_separation for symbol in self.element_list
                }
            else:
                for symbol in self.element_list:
                    if symbol not in single_species.min_separation.keys():
                        single_species.min_separation[symbol] = global_separation

    def get_species(self, symbol: str) -> Species:
        """
        Gets the `Species` object for a given chemical symbol.

        Returns
        -------
        Species

        Raises
        ------
        ValueError
            If `symbol` does not match any known `Species`.
        """
        try:
            i = self.element_list.index(symbol)
            return self.all_species[i]
        except ValueError as e:
            raise ValueError(
                "No atomic species with symbol `{0}` present in `{1}`."
                "".format(symbol, self.element_list)
            ) from e


class AllStructures(Dict[str, Structure]):
    """
    Dictionary with Structures as values, and their names as keys.

    Parameters
    ----------
    structures: one or more Structure
        A number of Structure objects that make up the entirety of the reference data set.
        These should all have the same elements and masses, but may have differences in how the
        active learning is handled, for example different energy and force tolerances can be set
        for each. They must have distinct names.
    """

    def __init__(self, *structures: Structure):
        if len(structures) == 0:
            raise ValueError("At least one `Structure` object must be passed.")
        super().__init__()
        for structure in structures:
            if structure.name in self.keys():
                raise ValueError(
                    "Cannot have multiple structures with the name `{}`"
                    "".format(structure.name)
                )
            self[structure.name] = structure

            if structure.element_list != structures[0].element_list:
                raise ValueError("All structures must have the same `element_list`")

            if structure.atomic_number_list != structures[0].atomic_number_list:
                raise ValueError(
                    "All structures must have the same `atomic_number_list`"
                )

            if structure.mass_list != structures[0].mass_list:
                raise ValueError("All structures must have the same `mass_list`")

        self.element_list = structures[0].element_list
        self.atomic_number_list = structures[0].atomic_number_list
        self.mass_list = structures[0].mass_list


class Frame(Atoms):
    """
    Holds information for a single frame of a `Structure`.

    Parameters
    ----------
    lattice: np.ndarray
        Array of float with shape (3, 3) representing the lattice vectors of the frame.
    symbols: np.ndarray
        Array of str with shape (N), where N is the number of atoms in the frame,
        representing the chemical symbols of each atom.
    positions: np.ndarray
        Array of float with shape (N, 3), where N is the number of atoms in the frame,
        representing the positions of each atom.
    charges: np.ndarray = None
        Array of float with shape (N), where N is the number of atoms in the frame,
        representing the charge of each atom. Optional, default is None.
    forces: np.ndarray = None
        Array of float with shape (N, 3), where N is the number of atoms in the frame,
        representing the forces of each atom. Optional, default is None.
    energy: float = None
        float with shape representing the total energy of the frame. Optional, default is None.
    name: str = None
        The name of the Structure represented by this Frame. Optional, default is None.
    """

    def __init__(
        self,
        lattice: np.ndarray,
        symbols: np.ndarray,
        positions: np.ndarray,
        charges: np.ndarray = None,
        forces: np.ndarray = None,
        energy: float = None,
        name: str = None,
    ):
        super().__init__(
            symbols=symbols, positions=positions, cell=lattice, charges=charges
        )
        self.forces = forces
        self.energy = energy
        self.name = name

    @property
    def symbols(self) -> np.ndarray:
        """
        Get the chemical symbols of all constituent atoms, in order.

        Returns
        -------
        ndarray of str
        """
        return np.array(super().symbols)

    @property
    def lattice(self) -> np.ndarray:
        """
        Get the the lattice vectors as a (3, 3) array.

        Returns
        -------
        ndarray of float
        """
        return np.array(super().cell)

    @property
    def charges(self) -> np.ndarray:
        """
        Get the charges of all constituent atoms, in order.

        Returns
        -------
        ndarray of float
        """
        return np.array(super().get_initial_charges())

    def check_nearest_neighbours(
        self,
        element_i: str,
        element_j: str,
        d_min: float,
    ) -> Tuple[bool, float]:
        """
        Checks all positions of `element_i` against those of `element_j` for whether they
        satisfy the nearest neighbour constraint `d_min`.

        Parameters
        ----------
        element_i : str
            Chemical symbol of the first element to consider.
        element_j : str
            Chemical symbol of the second element to consider.
        d_min : float
            The minimum seperatation allowed for the positions of elements i and j.

        Returns
        -------
        bool, float
            First element is whether the positions satisfy the minimum seperation criteria.
            Second element is the seperation that caused rejection, or -1 in the case of
            acceptance.
        """
        lat = self.lattice
        pos_i = self.positions[self.symbols == element_i]
        pos_j = self.positions[self.symbols == element_j]
        if len(pos_i) == 0 or len(pos_j) == 0:
            return True, -1

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
        if element_i == element_j:
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

    def check_min_separation(self, structure: Structure) -> bool:
        """
        Checks the interatomic distances for atom in the structure to ensure
        that none are within the minimum requried seperation of `structure`.

        Parameters
        ----------
        structure : Structure
            The Structure that this Frame corresponds to.

        Returns
        -------
        bool
            Whether the arguments were accepted as a valid structure.
        """
        for i, element_i in enumerate(structure.element_list):
            for element_j in structure.element_list[i:]:
                accepted, d = self.check_nearest_neighbours(
                    element_i,
                    element_j,
                    structure.get_species(element_i).min_separation[element_j],
                )
                if not accepted:
                    print(
                        "Too small interatomic distance between {}-{}: {} Ang".format(
                            element_i, element_j, d
                        )
                    )
                    return False

        return True


class Dataset(List[Frame]):
    """
    Holds a series of `Frame` objects representing a dataset. If `all_structures` is provided
    then will attempt to associate each Frame with a Structure name based on the comments in
    the data file, or if there is only one Structure in `all_structures` then all frames will
    be associated with that Structure. This is needed in order to run minimum separation
    checks.

    Parameters
    ----------
    data_file: str
        Complete file path of the n2p2 data file to read.
    all_structures: AllStructures = None
        Representations of all Structures in the Dataset. Optional, default is None.
    """

    def __init__(
        self,
        data_file: str,
        all_structures: AllStructures = None,
    ):
        self.all_structures = all_structures
        super().__init__(self.read_data_file(data_file=data_file))

    @property
    def n_atoms_per_frame(self) -> np.ndarray:
        """
        Get the number of atoms in each frame.

        Returns
        -------
        ndarray of int
        """
        return np.array([len(frame) for frame in self])

    def read_data_file(
        self,
        data_file: str,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
        """
        Read n2p2 structure file and return the data as python objects.

        Parameters
        ----------
        data_file: str
            The complete n2p2 structure file path to read from.

        Returns
        -------
        list of tuple of `ndarray` and float
            Each tuple within the outer list represents a single frame from the structure file.
            Within the tuple, the elements are as follows:
            - ndarray of float with shape (3, 3) representing the lattice vectors
            - ndarray of str with length N, where N is the number of atoms in the frame,
                representing the chemical species of the atom
            - ndarray of float with shape (N, 3), where N is the number of atoms in the frame,
                representing the position vector of each atom
            - ndarray of float with shape (N, 3), where N is the number of atoms in the frame,
                representing the force vector of each atom
            - float of the frames total energy
        """
        with open(data_file) as f:
            lines = f.readlines()

        frames = []
        lattice = []
        elements = []
        positions = []
        forces = []
        energy = None
        for line in lines:
            words = line.split()
            if line.strip() == "begin":
                lattice = []
                elements = []
                positions = []
                forces = []
                energy = None
                if self.all_structures is not None and len(self.all_structures) == 1:
                    name = list(self.all_structures)[0]
                else:
                    name = None
            elif words[0] == "comment":
                if (
                    (words[1] == "structure")
                    and (self.all_structures is not None)
                    and (words[2] in self.all_structures)
                ):
                    name = words[2]
            elif words[0] == "lattice":
                lattice.append(np.array([float(words[j]) for j in (1, 2, 3)]))
            elif words[0] == "atom":
                elements.append(words[4])
                positions.append(
                    [
                        float(words[1]),
                        float(words[2]),
                        float(words[3]),
                    ]
                )
                forces.append(
                    [
                        float(words[-3]),
                        float(words[-2]),
                        float(words[-1]),
                    ]
                )
            elif words[0] == "energy":
                energy = float(words[1])
            elif line.strip() == "end":
                frames.append(
                    Frame(
                        lattice=np.array(lattice),
                        symbols=np.array(elements),
                        positions=np.array(positions),
                        forces=np.array(forces),
                        energy=energy,
                        name=name,
                    )
                )

        return frames

    def check_min_separation_all(self) -> Iterator[bool]:
        """
        Checks all Frames in the Dataset to ensure they satisfy the minimum separation of
        `self.all_structures`. Will raise a ValueError if `self.all_structures` was not set.

        Yields
        ------
        Iterator[bool]
            Whether each frame satisfies the separation for the particular Structure it
            represents.
        """
        if self.all_structures is None:
            raise ValueError("Cannot check separation if `all_structures` is not set.")

        for frame in self:
            structure = self.all_structures[frame.name]
            yield frame.check_min_separation(structure=structure)
