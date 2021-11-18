"""
Reads n2p2 atomic-env.G files for information used in the workflow.
"""

from typing import Dict, List

import numpy as np


def read_atomenv(
    atomenv_name: str, elements: List[str], atoms_per_frame: int, dtype: str = "float32"
) -> Dict[str, np.ndarray]:
    """
    Parameters
    ----------
    atomenv_name: str
        Complete file path of the n2p2 "atomic-env.G" file.
    elements: List[str]
        List of str representing the chemical symbol of all atoms in the structure.
    atoms_per_frame: int
        Number of atoms in each frame.
    dtype: str, optional
        numpy data type to use in the returned values. Default is "float32".

    Returns
    -------
    dict of str ndarray
        Each key is a str representing the chemical symbol of an element present in the
        structure. The corresponding value is an array of `dtype` with the shape (N, M, L)
        where N is the number of frames, M is the number of atoms per frame, and L is the
        number of symmetry functions.
    """
    element_environments = {element: [] for element in elements}
    with open(atomenv_name) as f:
        atom_index = 0
        frame_index = 0
        line = f.readline()
        while line:
            if atom_index == 0:
                for element in elements:
                    element_environments[element].append([])

            data = line.split()
            element_environments[data[0]][-1].append(data[1:])

            atom_index += 1
            line = f.readline()

            if atom_index == atoms_per_frame:
                atom_index = 0
                frame_index += 1

    return {
        key: np.array(value).astype(dtype)
        for key, value in element_environments.items()
    }
