"""
Abstract class for performing data selection using pre-calculated atomic environments.
"""

from os.path import join
import time
from typing import List

import numpy as np


from cc_hdnnp.data import Data
from cc_hdnnp.dataset import Dataset
from cc_hdnnp.file_operations import read_atomenv


class DataSelector:
    """
    Abstract class for performing data selection using pre-calculated atomic environments.

    Parameters
    ----------
    data_controller: Data,
        Controller object used to get the n2p2 directory to work from, and elements present.
    n2p2_directory_index: int = 0,
        Used in conjunction with `data_controller` to get the n2p2 directory to work from.
        Default is 0.
    verbosity: int = 1,
        How much information to print during operation. 0 will result in no printing,
        1 prints summaries and 2 prints information at every step of the CUR decomposition.
        Default is 1.
    dtype: str = "float32",
        Data type to use in the numpy arrays. Default is "float32".
    """

    def __init__(
        self,
        data_controller: Data,
        n2p2_directory_index: int = 0,
        verbosity: int = 1,
        dtype: str = "float32",
    ):
        self.dtype = dtype
        self.verbosity = verbosity
        self.elements = data_controller.elements
        self.n2p2_directory = data_controller.n2p2_directories[n2p2_directory_index]
        t1 = time.time()
        dataset = Dataset(data_file=join(self.n2p2_directory, "input.data"))
        if len(np.unique(dataset.n_atoms_per_frame)) > 1:
            raise ValueError(
                "Datasets containing a varying number of atoms per frame are not supported, "
                "datset contained {} atoms in different frames."
                "".format(np.unique(dataset.n_atoms_per_frame))
            )
        if len(np.unique(dataset.n_atoms_per_frame)) == 0:
            raise ValueError("Dataset contains no frames.")

        self._atom_environments = read_atomenv(
            join(self.n2p2_directory, "atomic-env.G"),
            data_controller.elements,
            atoms_per_frame=dataset.n_atoms_per_frame[0],
            dtype=dtype,
        )
        t2 = time.time()
        if verbosity >= 1:
            print("Values read from file in {} s".format(t2 - t1))

    @property
    def n_frames(self) -> int:
        """
        Returns
        -------
        int
            The number of frames present in the environments for the elements.
            If they have a different number, a ValueError is raised.
        """
        n = None
        for environment in self._atom_environments.values():
            if n is None:
                n = environment.shape[0]
            elif n != environment.shape[0]:
                raise ValueError(
                    "Not all elements have the same number of frames ({}, {})"
                    "".format(n, environment.shape[0])
                )
        return n

    @property
    def n_atoms_list(self) -> List[int]:
        """
        Returns
        -------
        List[int]
            The number of atoms for each element, in the same order as `self.elements`.
        """
        return [e.shape[1] for e in self._atom_environments.values()]

    @property
    def n_symf_list(self) -> List[int]:
        """
        Returns
        -------
        List[int]
            The number of symmetry functions for each element,
            in the same order as `self.elements`.
        """
        return [e.shape[2] for e in self._atom_environments.values()]

    def get_environments(self, element: str) -> np.ndarray:
        """
        Parameters
        ----------
        element: str
            The chemical symbol of the element to return environments for.
        Returns
        -------
        np.ndarray
            An array of float with shape (N, M, L) where N is the number of frames,
            M is the number of atoms and L is the number of symmetry functions.
        """
        return np.copy(self._atom_environments[element])
