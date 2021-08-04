"""
Class for containing information about atomic species in the structure of interest.
"""


from typing import Dict, List


class Species:
    """
    Holds information about a single atomic species.

    Attributes
    ----------
    symbol : str
        Chemical symbol
    atomic_number : int
        Atomic number
    mass : float
        Mass of the species in AMU.
    min_separation : dict of [str, float], optional
        The minimum allowed distance between atoms of this species and those of other species.
        The keys should be the chemical symbols of atomic species (including this one), with the
        values being separation in Bohr. Default is `None`.
    """

    def __init__(
        self,
        symbol: str,
        atomic_number: int,
        mass: float,
        min_separation: Dict[str, float] = None,
    ):
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.mass = mass
        self.min_separation = min_separation


class AllSpecies:
    """
    Holds information about all atomic species in the structure.

    Attributes
    ----------
    species_list : list of Species
        All the species that make up the structure.
    element_list : list of str
        The chemical symbols of all structures that make up the structure.
    """

    def __init__(self, *species: Species, global_separation: float = 1.0):
        self.species_list = list(species)
        self.species_list.sort(key=lambda x: x.atomic_number)
        for single_species in self.species_list:
            if single_species.min_separation is None:
                # If `min_separation` is not set, use the global value for all species
                single_species.min_separation = {
                    symbol: global_separation for symbol in self.element_list
                }
            else:
                for symbol in self.element_list:
                    if symbol not in single_species.min_separation.keys():
                        single_species.min_separation[symbol] = global_separation

    @property
    def element_list(self):
        """
        Get the symbols of all species, sorted by atomic number.

        Returns
        -------
        list of str
        """
        return [single_species.symbol for single_species in self.species_list]

    @property
    def mass_list(self):
        """
        Get the masses of all species, sorted by atomic number.

        Returns
        -------
        list of str
        """
        return [single_species.mass for single_species in self.species_list]

    def get_species(self, symbol: str) -> Species:
        """"""
        try:
            i = self.element_list.index(symbol)
            return self.species_list[i]
        except ValueError as e:
            raise ValueError(
                "No atomic species with symbol `{0}` present in `{1}`."
                "".format(symbol, self.species_list)
            ) from e


class Structure:
    """
    Holds information about a given structure in the reference data.

    # TODO

    Attributes
    ----------
    all_species : AllSpecies
        Object detailing the species that make up the structure.
    """

    def __init__(
        self,
        name: str,
        all_species: AllSpecies,
        delta_E: float,
        delta_F: float,
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

        # Leave timestep separation variables as None, as a sensible default value will depend on
        # the total length of the simulation being run.
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


class AllStructures:
    """
    Holds information about structures.

    # TODO
    """

    def __init__(self, *structures: Structure):
        self.structure_dict: Dict[str, Structure] = {}
        for structure in structures:
            if structure.name in self.structure_dict.keys():
                raise ValueError(
                    "Cannot have multiple structures with the name `{}`"
                    "".format(structure.name)
                )
            self.structure_dict[structure.name] = structure

            if (
                structure.all_species.element_list
                != structures[0].all_species.element_list
            ):
                raise ValueError("All structures must have the same `element_list`")

            if structure.all_species.mass_list != structures[0].all_species.mass_list:
                raise ValueError("All structures must have the same `mass_list`")

        self.element_list = structures[0].all_species.element_list
        self.mass_list = structures[0].all_species.mass_list
