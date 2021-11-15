"""
Class for containing information about atomic species in the structure of interest.
"""


from typing import Dict, List

from ase.data import atomic_masses, atomic_numbers, chemical_symbols


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
