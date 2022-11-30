"""
Class for containing information about atomic species in the structure of interest.
"""


from ast import literal_eval
from copy import deepcopy
from typing import Dict, Iterable, Iterator, List, Tuple

from ase.atoms import Atoms
from ase.geometry import is_orthorhombic
from ase.io.formats import read, write
import numpy as np
import warnings

from cc_hdnnp.structure import AllStructures, Structure
from .units import UNITS


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
    units: Dict[str, str] = None
        The units to assign to the values given. Supports keys of "length" and "energy".
        If None, then the ASE defaults of "Ang" and "eV" will be assumed.
    statistics: List[str] = None
        Statistics of extrapolations associated with this structure from the AL process.
        Optional, default is None.
    """

    def __init__(
        self,
        lattice: np.ndarray,
        symbols: np.ndarray,
        positions: np.ndarray,
        charges: np.ndarray = None,
        forces: np.ndarray = None,
        energy: float = 0.0,
        name: str = None,
        frame_file: str = None,
        units: Dict[str, str] = None,
        statistics: List[str] = None,
        pbc: List[bool] = [True,True,True],
    ):
        super().__init__(
            symbols=symbols, positions=positions, cell=lattice, charges=charges
        )
        if forces is None:
            self.forces = np.zeros(shape=positions.shape)
        else:
            self.forces = forces
        self.energy = energy
        self.name = name
        self.frame_file = frame_file
        self.statistics = statistics
        self.set_pbc(pbc)
        self._units = {"energy": "eV", "length": "Ang"}
        if units is not None:
            self._set_units(units=units)

    def _set_units(self, units: Dict[str, str]):
        """
        Sets `self._units`.

        Parameters
        ----------
        units: Dict[str, str]
            The new units to assign. Supports keys of "length" and "energy".
        """
        if "energy" in units:
            self._units["energy"] = units["energy"]

        if "length" in units:
            self._units["length"] = units["length"]

        for key in units:
            if key not in ["energy", "length"]:
                raise ValueError(
                    "Only 'energy', 'length' are supported keys in `units`, "
                    "but {} was provided.".format(units)
                )

    @property
    def units(self) -> Dict[str, str]:
        """
        Returns the units that are currently in use.

        Returns
        -------
        Dict[str, str]
        """
        return self._units

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
    def is_orthorhombic(self) -> bool:
        """
        Whether the lattice of the Frame is orthorhombic or not.

        Returns
        -------
        bool
        """
        return is_orthorhombic(super().cell)

    @property
    def charges(self) -> np.ndarray:
        """
        Get the charges of all constituent atoms, in order.

        Returns
        -------
        ndarray of float
        """
        return np.array(super().get_initial_charges())

    @property
    def n2p2_text(self) -> str:
        """
        Returns
        -------
        str
            The information of the Frame formatted into a string that can be read by n2p2.
        """
        text = "begin\n"
        if self.name is not None:
            text += "comment structure {}\n".format(self.name)
        if self.frame_file is not None:
            text += "comment file {}\n".format(self.frame_file)
        if self.statistics is not None and len(self.statistics) == 4:
            text += "comment statistics {}\n".format("; ".join(self.statistics))
        text += "comment units {}\n".format(self.units)
        for vector in self.lattice:
            text += "lattice {:16.8f} {:16.8f} {:16.8f}\n".format(*vector)
        for i, symbol in enumerate(self.symbols):
            text += (
                "atom {:16.8f} {:16.8f} {:16.8f} {:4s} {:16.8f} 0.0000000 "
                "{:16.8f} {:16.8f} {:16.8f}\n".format(
                    *self.positions[i], symbol, self.charges[i], *self.forces[i]
                )
            )
        text += "energy {:16.8f}\n".format(self.energy)
        text += "charge {:16.8f}\n".format(np.sum(self.charges))
        text += "end\n"

        return text

    def change_units(self, new_units: Dict[str, str]):
        """
        Converts all lengths, energies and forces from their current units to those
        provided as an argument.

        Parameters
        ----------
        new_units: Dict[str, str]
            The new units to convert the frame to. Supports keys of "length" and "energy".
        """
        length_conversion = UNITS[self.units["length"]] / UNITS[new_units["length"]]
        energy_conversion = UNITS[self.units["energy"]] / UNITS[new_units["energy"]]
        force_conversion = energy_conversion / length_conversion

        self.positions *= length_conversion
        self.cell *= length_conversion
        self.energy *= energy_conversion
        self.forces *= force_conversion

        self._set_units(units=new_units)

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

    def check_threshold(
        self, energy_threshold: Tuple[float, float], force_threshold: float
    ) -> bool:
        """
        Checks the energy and force values to ensure all are within the provided
        thesholds.

        Parameters
        ----------
        energy_threshold : float or tuple of float
            The first and second entries are taken as the lower/upper bounds on energy.
        force_threshold : float
            Maximum absolute force value tolerated.

        Returns
        -------
        bool
            Whether the arguments were accepted as a valid structure.
        """
        if self.energy < energy_threshold[0] or self.energy > energy_threshold[1]:
            return False
        elif np.any(np.absolute(self.forces) > force_threshold):
            return False
        else:
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
    frames: List[Frame] = None
        A list of Frame objects used in addition to those loaded from file
        (if `data_file` is not None). Optional, default is None.
    data_file: str = None
        Complete file path of the data file to read. Optional, default is None.
    all_structures: AllStructures = None
        Representations of all Structures in the Dataset. Optional, default is None.
    format: str = "n2p2"
        The format of the `data_file` given. Should either be "n2p2"
        or a format recognised by ASE. Default is "n2p2".
    verbosity: int = 0,
        How much information to print during reading. 0 will result in no printing,
        1 prints warnings. Optional, default is 0.
    units: Dict[str, str]
        The units to assign to the values given. Supports keys of "length" and "energy".
        Unused if `format` is not "n2p2", or if file contains comments with "units".
        Optional, default is None.
    **kwargs
        If `format` is not "n2p2", any other keyword arguments will be passed to the ASE
        read function.
    """

    def __init__(
        self,
        frames: List[Frame] = None,
        data_file: str = None,
        all_structures: AllStructures = None,
        format: str = "n2p2",
        verbosity: int = 0,
        units: Dict[str, str] = None,
        **kwargs,
    ):
        self.all_structures = all_structures
        self.verbosity = verbosity

        if frames is not None:
            for frame in frames:
                if not isinstance(frame, Frame):
                    raise TypeError(
                        "`frames` must be a list of Frame objects, "
                        "but had entries with type {}".format(type(frame))
                    )
        else:
            frames = []

        if data_file is None:
            super().__init__(frames)
        else:
            # Load from file
            if format == "n2p2":
                for frame in self.read_data_file(data_file=data_file, units=units):
                    frames.append(frame)
            else:
                if all_structures is not None and len(all_structures) == 1:
                    structure_name = list(all_structures.keys())[0]
                else:
                    structure_name = None
                for atoms in read(
                    filename=data_file, format=format, index=":", **kwargs
                ):
                    try:
                        frames.append(
                            Frame(
                                pbc = atoms.get_pbc(),
                                lattice=atoms.cell,
                                symbols=atoms.symbols,
                                positions=atoms.positions,
                                charges=atoms.get_initial_charges(),
                                name=structure_name,
                                forces=atoms.get_forces(),
                                energy=atoms.get_potential_energy(),
                                units=units,
                            )
                        )
                    except:
                        frames.append(
                            Frame(
                                pbc = atoms.get_pbc(),
                                lattice=atoms.cell,
                                symbols=atoms.symbols,
                                positions=atoms.positions,
                                charges=atoms.get_initial_charges(),
                                name=structure_name,
                                units=units,
                            )
                        )
            super().__init__(frames)

    @property
    def n_atoms_per_frame(self) -> np.ndarray:
        """
        Get the number of atoms in each frame.

        Returns
        -------
        ndarray of int
        """
        return np.array([len(frame) for frame in self])

    @property
    def all_names(self) -> np.ndarray:
        """
        Get the name of each the Structure each Frame represents, in order.

        Returns
        -------
        ndarray of str
            Length is N, the number of frames.
        """
        return np.array([frame.name for frame in self])

    @property
    def all_energies(self) -> np.ndarray:
        """
        Get the energy of each constituent frame, in order.

        Returns
        -------
        ndarray of float
            Length is N, the number of frames.
        """
        return np.array([frame.energy for frame in self])

    @property
    def all_forces(self) -> np.ndarray:
        """
        Get the forces of each atom in each constituent frame, in order.

        Returns
        -------
        ndarray of float
            Shape is (N, M, 3) where N is number of frames and M is number of atoms per frame.
        """
        return np.array([frame.forces for frame in self])

    @property
    def all_lattices(self) -> np.ndarray:
        """
        Get the lattice vectors of each constituent frame, in order.

        Returns
        -------
        ndarray of float
            Shape is (N, 3, 3) where N is number of frames.
        """
        return np.array([frame.lattice for frame in self])

    @property
    def all_symbols(self) -> np.ndarray:
        """
        Get the chemical symbols of atoms in each constituent frame, in order.

        Returns
        -------
        ndarray of str
            Shape is (N, M) where N is number of frames, M is number of atoms.
        """
        return np.array([frame.symbols for frame in self])

    @property
    def all_positions(self) -> np.ndarray:
        """
        Get the positons of each atom in each constituent frame, in order.

        Returns
        -------
        ndarray of float
            Shape is (N, M, 3) where N is number of frames, M is number of atoms.
        """
        return np.array([frame.positions for frame in self])

    @property
    def all_charges(self) -> np.ndarray:
        """
        Get the charges of each atom in each constituent frame, in order.

        Returns
        -------
        ndarray of float
            Shape is (N, M) where N is number of frames, M is number of atoms.
        """
        return np.array([frame.charges for frame in self])

    @property
    def all_statistics(self) -> np.ndarray:
        """
        Get the extrapolation statistics of each constituent frame, in order.

        Returns
        -------
        ndarray of str
            Shape is (N,) where N is number of frames.
        """
        return np.array([frame.statistics for frame in self], dtype=object)

    def read_data_file(
        self,
        data_file: str,
        units: Dict[str, str] = None,
    ) -> List[Frame]:
        """
        Read n2p2 structure file and return the data as a list of Frames.

        Parameters
        ----------
        data_file: str
            The complete n2p2 structure file path to read from.

        Returns
        -------
        List[Frame]
            Each individual structure in the dataset represented as a Frame, in order.
        """
        with open(data_file) as f:
            lines = f.readlines()

        frames = []
        lattice = []
        elements = []
        positions = []
        forces = []
        charges = []
        statistics = []
        energy = None
        frame_file = None
        for line in lines:
            words = line.split()
            if line.strip() == "begin":
                lattice = []
                elements = []
                positions = []
                forces = []
                charges = []
                statistics = []
                energy = None
                if self.all_structures is not None and len(self.all_structures) == 1:
                    name = list(self.all_structures)[0]
                else:
                    name = None

            elif words[0] == "comment":
                if len(words) >= 3 and (words[1] == "structure"):
                    name = words[2]

                if len(words) >= 3 and words[1] == "units":
                    if units is None:
                        units = literal_eval("".join(words[2:]))
                    elif literal_eval("".join(words[2:])) != units:
                        warnings.warn(
                            f"The units {units} were given when initialising the "
                            f'dataset, but units {literal_eval("".join(words[2:]))} '
                            f"are specified in comments in the datafile. "
                            f"Using units: {units}. Please ensure these are correct."
                        )
                elif len(words) >= 4 and words[3].startswith("units="):
                    if units is None:
                        units = literal_eval(words[3][6:])
                    elif literal_eval(words[3][6:]) != units:
                        warnings.warn(
                            f"The units {units} were given when initialising the "
                            f'dataset, but units {literal_eval(words[3][6:])} '
                            f"are specified in comments in the datafile. "
                            f"Using units: {units}. Please ensure these are correct."
                        )

                if len(words) >= 3 and words[1] == "statistics":
                    statistics_text = " ".join(words[2:])
                    statistics = statistics_text.split("; ")

                if len(words) >= 3 and words[1] == "file":
                    frame_file = words[2]

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
                charges.append(float(words[5]))
            elif words[0] == "energy":
                energy = float(words[1])
            elif words[0] == "charge":
                if self.verbosity > 0 and not np.isclose(
                    np.sum(charges), float(words[1])
                ):
                    print(
                        "WARNING: Total charge {} and sum of atomic charge {} are not close"
                        "".format(float(words[1]), np.sum(charges))
                    )
            elif line.strip() == "end":
                if len(lattice) == 0:
                    lattice = None
                else:
                    lattice = np.array(lattice)
                # Assume the n2p2 data file uses Ha and Bohr,
                # given these are the RuNNer defaults
                if units is None:
                    units = {"energy": "Ha", "length": "Bohr"}
                frames.append(
                    Frame(
                        lattice=lattice,
                        symbols=np.array(elements),
                        positions=np.array(positions),
                        forces=np.array(forces),
                        charges=np.array(charges),
                        energy=energy,
                        name=name,
                        units=units,
                        statistics=statistics,
                        frame_file=frame_file,
                    )
                )

        return frames

    def write_n2p2_file(
        self,
        file_out: str,
        conditions: Iterable[bool] = None,
        append: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """
        Write the Dataset to file in the n2p2 format.

        Parameters
        ----------
        file_out: str
            The complete filepath to write the dataset to.
        conditions: Iterable[bool] = None
            An iterable object of bool, if True then the corresponding Frame in the Dataset
            will be written to file, else it will be omitted (but still be present in `self`).
        append: bool = False
            Whether to overwrite or append to `file_out`. Optional, default is False.

        Returns
        -------
        Tuple[List[int], List[int]]
            First entry is a list of the indices of Frames that were written to file.
            Second entry is a list of the indices of Frames that were NOT written to file.
        """
        selected = []
        removed = []
        mode = "a" if append else "w"
        with open(file_out, mode=mode) as f:
            if conditions is None:
                conditions = (True for _ in self)
            for i, condition in enumerate(conditions):
                if condition:
                    f.write(self[i].n2p2_text)
                    selected.append(i)
                else:
                    removed.append(i)

        return selected, removed

    def write_extxyz_file(
        self,
        file_out: str,
        conditions:Iterable[bool] = None,
        append: bool = False,
        units: Dict[str, str] = {"energy": "eV", "length": "Ang"},
    ):
        """
        Write the Dataset to file in the extxyz format.

        Parameters
        ----------
        file_out: str
            The complete filepath to write the dataset to.
        units: Dict[str, str] = None
            The units to assign to the values given. Supports keys of "length" and "energy".
            Default is `{"energy": "eV", "length": "Ang"}`
        conditions: Iterable[bool] = None
            An iterable object of bool, if True then the corresponding Frame in the Dataset
            will be written to file, else it will be omitted (but still be present in `self`).
        append: bool = False
            Whether to overwrite or append to `file_out`. Optional, default is False.

        Returns
        -------
        Tuple[List[int], List[int]]
            First entry is a list of the indices of Frames that were written to file.
            Second entry is a list of the indices of Frames that were NOT written to file.
        """
        output = ""
        self.change_units_all(units)
        selected = []
        removed = []
        if conditions is None:
            conditions = (True for _ in self)
        for i, condition in enumerate(conditions):
            if condition:
                selected.append(i)
                output += (
                    f'{len(self[i])}\n'

                    f'Lattice="{self[i].lattice[0,0]} {self[i].lattice[0,1]} '
                    f'{self[i].lattice[0,2]} {self[i].lattice[1,0]} {self[i].lattice[1,1]} '
                    f'{self[i].lattice[1,2]} {self[i].lattice[2,0]} {self[i].lattice[2,1]} '
                    f'{self[i].lattice[2,2]}" '

                    f'Properties=species:S:1:pos:R:3:forces:R:3 '
                    f'energy={self[i].energy} pbc="{self[i].pbc[0]} '
                    f'{self[i].pbc[1]} {self[i].pbc[2]}"\n'
                )
                for s, X, F in zip(self[i].symbols, self[i].positions, self[i].forces):
                    output += (
                        f'{s}    {X[0]}    {X[1]}    {X[2]}    '
                        f'{F[0]}    {F[1]}    {F[2]}\n'
                    )
            else:
                removed.append(i)
        mode = "a" if append else "w"
        with open(file_out, mode=mode) as f:
            f.write(output)

        return selected, removed

    def write(
        self,
        file_out: str,
        format: str = "n2p2",
        conditions: Iterable[bool] = None,
        append: bool = False,
        units : Dict[str, str] = None,
        **kwargs,
    ) -> Tuple[List[int], List[int]]:
        """
        Write the Dataset to file.

        Parameters
        ----------
        file_out: str
            The complete filepath to write the dataset to.
        format: str = "n2p2"
            The format to use when writing to file.
            Should either be "n2p2" or a format supported by ASE. Default is "n2p2".
        conditions: Iterable[bool] = None
            An iterable object of bool, if True then the corresponding Frame in the Dataset
            will be written to file, else it will be omitted (but still be present in `self`).
        append: bool = False
            Whether to overwrite or append to `file_out`. Optional, default is False.
        **kwargs
            If `format` is not "n2p2", any other keyword arguments will be passed to the ASE
            read function.

        Returns
        -------
        Tuple[List[int], List[int]]
            First entry is a list of the indices of Frames that were written to file.
            Second entry is a list of the indices of Frames that were NOT written to file.
        """
        if units is None and format != "n2p2":
            units = {"energy": "eV", "length": "Ang"}
        if units is not None:
            self.change_units_all(units)
        if format == "n2p2":
            return self.write_n2p2_file(
                file_out=file_out, conditions=conditions, append=append
            )
        elif format == "extxyz":
            return self.write_extxyz_file(
                file_out=file_out, conditions=conditions, append=append
            )
        else:
            selected = []
            removed = []
            if conditions is not None:
                images = []
                for i, condition in enumerate(conditions):
                    if condition:
                        images.append(self[i])
                        selected.append(i)
                    else:
                        removed.append(i)
            else:
                images = self
                selected = list(range(len(self)))
            write(
                filename=file_out, images=images, format=format, append=append, **kwargs
            )

            return selected, removed

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

    def check_threshold_all(
        self, energy_threshold: Tuple[float, float], force_threshold: float
    ) -> Iterator[bool]:
        """
        Checks all Frames in the Dataset to ensure they satisfy the energy and force
        theshold provided.

        Parameters
        ----------
        energy_threshold : float or tuple of float
            The first and second entries are taken as the lower/upper bounds on energy.
        force_threshold : float
            Maximum absolute force value tolerated.

        Yields
        ------
        Iterator[bool]
            Whether each frame satisfies the thresholds.
        """
        for frame in self:
            yield frame.check_threshold(
                energy_threshold=energy_threshold, force_threshold=force_threshold
            )

    def change_units_all(self, new_units: Dict[str, str]):
        """
        Change the units of all constituent Frames simultaneously.

        Parameters
        ----------
        new_units: Dict[str, str]
            The new units to convert the Dataset to. Supports keys of "length" and "energy".
        """
        for frame in self:
            frame.change_units(new_units=new_units)

    def filter_energies(
        self,
        bins: int = 8,
        samples: int = 25,
    ):
        """
        Returns a filter to reduce the number of frames, while ensuring a spread of energies.

        Parameters
        ----------
        bins: int
            Number of bins to divide the range of energies into. Default is 8.
        samples: int
            Estimated number of samples to take from each bin. Default is 25.
        """
        energies = self.all_energies
        E_min = np.min(energies)
        E_max = np.max(energies)

        bin_edges = []
        for i in range(bins + 1):
            bin_edges.append(E_min + (i * (E_max - E_min) / bins))

        sample_indicies = np.empty([0], dtype=int)

        for i in range(bins):
            energy_mask = np.logical_and(energies >= bin_edges[i], energies <= bin_edges[i+1])
            bin_indicies = np.where(energy_mask)[0]

            if len(bin_indicies) > samples:
                bin_indicies = np.random.choice(bin_indicies, samples, replace=False)

            sample_indicies = np.append(sample_indicies, bin_indicies)

        conditions=[i in sample_indicies for i, _ in enumerate(self)]
        return conditions

    def compare_structure(self, frame, permute=True):
        compare_atms = Atoms(positions=frame.get_positions(),symbols=frame.get_chemical_symbols(),cell=frame.get_cell(), pbc = [True,True,True]);
        dist = np.zeros(len(self))
        txt = ""
        for i, s1 in enumerate(self):
            self_atms = Atoms(positions=s1.get_positions(),symbols=s1.get_chemical_symbols(),cell=s1.get_cell(), pbc = [True,True,True]);
            dist[i] = distance(self_atms, compare_atms, permute)
            txt += str(dist[i]) + " "
        txt += '\n'
        print(txt)
        return dist

