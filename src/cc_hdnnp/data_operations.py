from copy import deepcopy
from typing import List, Tuple

import numpy as np

from cc_hdnnp.structure import Structure


def check_nearest_neighbours(
    lat: List[List[float]],
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    ii: bool,
    d_min: float,
) -> Tuple[bool, float]:
    """
    Checks all positions in `pos_i` against those of `pos_j`, and checks whether they
    satisfy the nearest neighbour constraint `d_min`.

    Parameters
    ----------
    lat : list of list of float
        A list with three entries, each another three entry list representing a lattice
        vector of the structure.
    pos_i : np.ndarray
        Array where the first dimension indexes the atoms of a particular element in the
        structure, and the second is length 3 representing the position of that atom in the
        cartesian co-ordinates.
    pos_j : np.ndarray
        Array where the first dimension indexes the atoms of a particular element in the
        structure, and the second is length 3 representing the position of that atom in the
        cartesian co-ordinates.
    ii : bool
        Whether the element of `pos_i` and `pos_j` is the same.
    d_min : float
        The minimum seperatation allowed for the positions of elements i and j.

    Returns
    -------
    bool, float
        First element is whether the positions satisfy the minimum seperation criteria.
        Second element is the seperation that caused rejection, or -1 in the case of
        acceptance.
    """
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


def check_structure(
    lattice: List[List[float]],
    element: np.ndarray,
    position: np.ndarray,
    structure: Structure,
) -> bool:
    """
    TODO UPDATE
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
    for i, element_i in enumerate(structure.all_species.element_list):
        for element_j in structure.all_species.element_list[i:]:
            accepted, d = check_nearest_neighbours(
                lattice,
                position[element == element_i],
                position[element == element_j],
                element_i == element_j,
                structure.all_species.get_species(element_i).min_separation[element_j],
            )
            if not accepted:
                return False

    return True
