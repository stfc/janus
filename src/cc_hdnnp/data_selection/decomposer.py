"""
Class and functions for performing data reduction by decomposition.
"""


from os.path import join
from shutil import copy
import time
from typing import Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD


from cc_hdnnp.dataset import Dataset
from .dataselector import DataSelector


class Decomposer(DataSelector):
    """
    Performs k=1 CUR decomposition in the manner described by Imbalzano et al. in
    https://arxiv.org/abs/1804.02150

    Parameters
    ----------
    data_controller: Controller
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

    def _validate_n_to_select(
        self,
        n_to_select_list: List[int],
        n_total: int,
        quantitiy: str,
        file_out_list: List[str],
    ) -> List[int]:
        """
        Checks the values of `n_to_select_list` is valid for `n_total` and `file_out_list`,
        and chooses a default value if `n_to_select_list` is None.

        Parameters
        ----------
        n_to_select_list: List[int],
            A list of how many of the features to select.
            If None it is set to [`n_total` / 2].
        n_total: int,
            The total number of features that can be selected from.
        quantitiy: str,
            The feature which is being selected. Only used to display information about the
            selection when printing.
        file_out_list: List[str],
            A list of the files to write to for each entry in `n_to_select`

        Returns
        -------
        List[int]
            `n_to_select_list` if it was set and valid, or [`n_total` / 2] otherwise.
        """
        if n_to_select_list is None:
            n_to_select_list = [n_total // 2]
            if self.verbosity >= 1:
                print(
                    "Selecting {} out of {} {}".format(
                        n_to_select_list[0], n_total, quantitiy
                    )
                )

        if len(n_to_select_list) != len(file_out_list):
            raise ValueError(
                "`n_to_select_list` and `file_out_list` must have the same lengths, "
                "but were {}, {}".format(len(n_to_select_list), len(file_out_list))
            )

        for n in n_to_select_list:
            if n >= n_total:
                raise ValueError(
                    "All entries in `n_to_select_list` must be less than the number of {}, "
                    "but they were {}, {}"
                    "".format(quantitiy, n_to_select_list, n_total)
                )
            elif n < 1:
                raise ValueError(
                    "All entries in `n_to_select_list` must be at least 1, but were {}"
                    "".format(n_to_select_list)
                )

        return n_to_select_list

    def _create_global_environment(self) -> np.ndarray:
        """
        Converts `self.atom_environments`, which are split by element and atom, into "global"
        environments representing a single frame. This is done by padding each element's
        symmetry function vector so they have a length of N_A + N_B + ... where N_A is the
        number of symmetry functions for element A and so on. In the resulting vector, the
        first N_A elements are reserved for element A, the next N_B for B and so on.
        Then environment for atoms of all elements in the frame are averaged into an
        environment for the frame.

        Returns
        -------
        np.ndarray
            An array of float with shape (N_A + N_B + ... , M) where N_A is the number of
            symmetry functions for element A, and M is the number of frames present in the
            data.
        """
        n_frames = self.n_frames
        global_environments = np.zeros(
            (
                n_frames,
                np.sum(self.n_atoms_list, dtype=int),
                np.sum(self.n_symf_list, dtype=int),
            ),
        )
        for i, element in enumerate(self.elements):
            environments = self.get_environments(element)
            environments_padded_atoms = np.concatenate(
                (
                    np.zeros(
                        (
                            n_frames,
                            np.sum(self.n_atoms_list[:i], dtype=int),
                            self.n_symf_list[i],
                        ),
                    ),
                    environments,
                    np.zeros(
                        (
                            n_frames,
                            np.sum(self.n_atoms_list[i + 1 :], dtype=int),
                            self.n_symf_list[i],
                        ),
                    ),
                ),
                axis=1,
            )
            global_environments += np.concatenate(
                (
                    np.zeros(
                        (
                            n_frames,
                            np.sum(self.n_atoms_list, dtype=int),
                            np.sum(self.n_symf_list[:i], dtype=int),
                        ),
                    ),
                    environments_padded_atoms,
                    np.zeros(
                        (
                            n_frames,
                            np.sum(self.n_atoms_list, dtype=int),
                            np.sum(self.n_symf_list[i + 1 :], dtype=int),
                        ),
                    ),
                ),
                axis=2,
            )

        mean_environment = np.mean(global_environments, axis=1)
        return mean_environment.T

    def _calculate_symf_weights(
        self,
        file_data: str,
        file_in: str,
        file_weights: str,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates importance weights for the symmetry functions, namely the estimated
        number of times they will be evaluated per frame. This is calculated as the number
        of atoms of the primary element times the average number of neighbouring atoms that
        lie within r_cutoff. This effectively preferentially selects functions what will
        rarely be evaluated due to reduced computational cost.

        Parameters
        ----------
        file_data: str,
            Complete file path to an "input.data" file, used to determine the average volume
            of the cell described by each frame.
        file_in: str,
            Complete file path to an "input.nn" file, used to determine the cutoff radius and
            elements involved in each symmetry function.
        file_weights: str,
            Complete file path to write the weights to. Not used in the calculation process,
            purely for debugging purposes.

        Returns
        -------
        Dict[str, np.ndarray]
            Each key corresponds to an element, with a value of an array containing the weight
            for each symmetry function.
        """
        dataset = Dataset(data_file=file_data)
        volume = np.mean([frame.get_volume() for frame in dataset])

        weights = {element: [] for element in self.elements}
        with open(file_in) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("symfunction_short"):
                element, type_int = line.split()[1:3]
                if type_int == "2":
                    element_1 = line.split()[3]
                    r_cutoff = float(line.split()[-1])
                    weights[element].append(
                        self.get_environments(element).shape[1]
                        * self.get_environments(element_1).shape[1]
                        * r_cutoff ** 3
                        * volume ** -1
                    )
                elif type_int in ("3", "9"):
                    element_1, element_2 = line.split()[3:5]
                    r_cutoff = float(line.split()[-2])
                    weights[element].append(
                        self.get_environments(element).shape[1]
                        * self.get_environments(element_1).shape[1]
                        * self.get_environments(element_2).shape[1]
                        * r_cutoff ** 6
                        * volume ** -2
                    )
                else:
                    raise ValueError(
                        "Weighting for symfunction type {} is not supported."
                        "".format(type_int)
                    )

        with open(file_weights, "w") as f:
            f.write(str(weights))

        weights = {e: np.array(w) for e, w in weights.items()}
        for element, element_weights in weights.items():
            if len(element_weights) != self.get_environments(element).shape[2]:
                raise ValueError(
                    "Should have {} weights for element {}, but had {}".format(
                        self.get_environments(element).shape[2],
                        element,
                        len(element_weights),
                    )
                )
            elif not np.all(np.isfinite(element_weights)):
                raise ValueError(
                    "Should have finite weights for element {}, but some were not".format(
                        element,
                    )
                )
            elif np.any(element_weights <= 0):
                raise ValueError(
                    "All weights should be positive for element {}, but some were not".format(
                        element,
                    )
                )

        return weights

    def _orthoganalise(
        self, environments_r2: np.ndarray, selected_index: int
    ) -> np.ndarray:
        """
        Orthoganalise `environments_r2` with respect to the `selected_index`.

        Parameters
        ----------
        environments_r2: np.ndarray
            Array of float with shape (N, M) where N is the number of samples and M the
            features for the CUR decomposition.
        selected_index: int
            The index of the feature selected from `environments_r2`.
        """
        selected_vector = environments_r2[:, selected_index : selected_index + 1]
        selected_magnitude_square = np.sum(selected_vector ** 2)
        projected_magnitudes_square = np.sum(
            environments_r2 * selected_vector, axis=0, keepdims=True
        )

        return environments_r2 - (
            selected_vector * projected_magnitudes_square / selected_magnitude_square
        )

    def _run_k1_CUR(
        self,
        environments_r2: np.ndarray,
        n_to_select: int,
        weights: np.ndarray,
    ) -> List[int]:
        """
        Runs the k=1 CUR selection on `environments_r2` until `n_to_select` features are
        chosen.

        Parameters
        ----------
        environments_r2: np.ndarray,
            A rank 2 - with shape (N, M) - array containing N samples of M features.
        n_to_select: int,
            The number of features, from a total of M, to select.
        weights: np.ndarray,
            Array of float with length M that the importance scores are divided by before
            selection.

        Returns
        -------
        List[int]
            The indices of the features which most represent the deviation across the dataset.
        """
        t1 = time.time()
        tsvd = TruncatedSVD(n_components=1)
        selected_indices = []
        while len(selected_indices) < n_to_select:
            finite_array = np.isfinite(environments_r2)
            if not np.all(finite_array):
                raise ValueError(
                    "Non-finite entries {} in array at position {}".format(
                        environments_r2[finite_array == False],  # noqa: E712
                        np.array(np.where(finite_array == False)),
                    )
                )
            tsvd.fit(environments_r2)
            right_vector = tsvd.components_[0, :]
            scores = right_vector ** 2
            if weights is not None:
                scores /= weights
            i = np.argmax(scores)
            selected_indices.append(i)
            if self.verbosity >= 2:
                print(
                    "Step {:3d}: selected {:6d} with a score of {:.6e}".format(
                        len(selected_indices), i, scores[i]
                    )
                )
            environments_r2 = self._orthoganalise(environments_r2, selected_index=i)
            environments_r2[:, selected_indices] = 0

        if self.verbosity >= 1:
            print(
                "Selected indices in {} s:\n{}".format(
                    time.time() - t1, selected_indices
                )
            )

        return selected_indices

    def run_CUR_symf(
        self,
        n_to_select_list: List[int] = None,
        weight: bool = True,
        file_in: str = "input.nn",
        file_out_list: List[str] = ("input.nn",),
        file_backup: str = "input.nn.CUR_backup",
        file_2d_atom: str = "CUR_2d_atom_{}.out",
        file_2d_frame: str = "CUR_2d_frame_{}.out",
    ):
        """
        Performs CUR decomposition selecting symmetry functions that when evaluated on the
        dataset best represent it.

        By providing multiple entries in `n_to_select_list` and `file_out_list`,
        multiple sets of symmetry functions can be obtained from a single CUR decomposition.

        Parameters
        ----------
        n_to_select_list: List[int] = None,
            List of how many features to select. Default is None, in which case it will be
            half of the total number of symmetry functions.
        weight: bool = True,
            Whether to divide the symmetry functions' CUR importance score by the estimated
            number of times they will be evaluated per frame. This is calculated as the number
            of atoms of the primary element times the average number of neighbouring atoms that
            lie within r_cutoff. This effectively preferentially selects functions what will
            rarely be evaluated due to reduced computational cost.
        file_in: str = None,
            Filepath relative to `self.n2p2 directory` to use as input.
            Default is "input.nn".
        file_out_list: str = None,
            List of filepaths relative to `self.n2p2 directory` to use as output for each of
            the values in `n_to_select_list`. Default is "input.nn".
        file_backup: str = None,
            Filepath relative to `self.n2p2 directory` to use as a backup for the original
            data. Default is "input.nn.CUR_backup".
        file_2d_atom: str = "CUR_2d_atom_{}.out"
            Filepath relative to `self.n2p2 directory` to write the environments for each atom
            in terms of their best two symmetry functions. Formatted with the chemical symbol
            for each element (as they all use distinct symmetry functions).
        file_2d_frame: str = "CUR_2d_frame_{}.out"
            Filepath relative to `self.n2p2 directory` to write the environments for each frame
            in terms of their best two symmetry functions. Formatted with the chemical symbol
            for each element (as they all use distinct symmetry functions).
        """
        if weight:
            weights = self._calculate_symf_weights(
                file_data=join(self.n2p2_directory, "input.data"),
                file_in=join(self.n2p2_directory, file_in),
                file_weights=join(self.n2p2_directory, "symf_weights.log"),
            )
        else:
            weights = {element: None for element in self.elements}

        all_selected_indices = {}
        for element in self.elements:
            if self.verbosity >= 1:
                print("\nElement: {}".format(element))

            environments = self.get_environments(element)
            n_frames, n_atoms, n_symf = environments.shape
            n_to_select_list = self._validate_n_to_select(
                n_to_select_list=n_to_select_list,
                n_total=n_symf,
                quantitiy="symmetry functions",
                file_out_list=file_out_list,
            )

            environments_r2 = environments.reshape((n_frames * n_atoms, n_symf))
            all_selected_indices[element] = self._run_k1_CUR(
                environments_r2=environments_r2,
                n_to_select=max(n_to_select_list),
                weights=weights[element],
            )

            # Write the environments to file expressed in terms of
            # their primary symmetry functions
            if len(all_selected_indices[element]) >= 2:
                index_0 = all_selected_indices[element][0]
                index_1 = all_selected_indices[element][1]
                file_atom_formatted = join(
                    self.n2p2_directory, file_2d_atom.format(element)
                )
                file_frame_formatted = join(
                    self.n2p2_directory, file_2d_frame.format(element)
                )
                with open(file_atom_formatted, "w") as f:
                    [
                        f.write("   {:3.9f}".format(v))
                        for v in environments_r2[:, index_0]
                    ]
                    f.write("\n")
                    [
                        f.write("   {:3.9f}".format(v))
                        for v in environments_r2[:, index_1]
                    ]
                with open(file_frame_formatted, "w") as f:
                    [
                        f.write("   {:3.9f}".format(v))
                        for v in np.mean(environments[:, :, index_0], axis=1)
                    ]
                    f.write("\n")
                    [
                        f.write("   {:3.9f}".format(v))
                        for v in np.mean(environments[:, :, index_1], axis=1)
                    ]
            elif self.verbosity >= 1:
                print(
                    "At least 2 symmetry functions must be selected "
                    "to write a 2D representation to file."
                )

        for i, n in enumerate(n_to_select_list):
            indices = {element: 0 for element in self.elements}
            copy(
                join(self.n2p2_directory, file_in),
                join(self.n2p2_directory, file_backup),
            )
            with open(join(self.n2p2_directory, file_in)) as f:
                lines = f.readlines()
            with open(join(self.n2p2_directory, file_out_list[i]), "w") as f:
                for line in lines:
                    if not line.startswith("symfunction_short"):
                        f.write(line)
                    else:
                        element = line.split()[1]
                        if indices[element] in all_selected_indices[element][:n]:
                            f.write(line)
                        else:
                            f.write("#" + line)
                        indices[element] += 1

    def run_CUR_data(
        self,
        n_to_select_list: List[int] = None,
        file_in: str = "input.data",
        file_out_list: List[str] = ("input.data",),
        file_backup: str = "input.data.CUR_backup",
    ):
        """
        Performs CUR decomposition selecting individual frames from the dataset
        that best represent the data as a whole.

        By providing multiple entries in `n_to_select_list` and `file_out_list`,
        multiple sets of frames can be obtained from a single CUR decomposition.

        Parameters
        ----------
        n_to_select_list: List[int] = None,
            List of how many features to select. Default is None, in which case it will be half
            of the total number of frames.
        file_in: str = None,
            File path relative to `self.n2p2 directory` to use as input.
            Default is "input.data".
        file_out_list: str = None,
            List of file paths relative to `self.n2p2 directory` to use as output for each of
            the values in `n_to_select_list`. Default is "input.nn".
        file_backup: str = None,
            File path relative to `self.n2p2 directory` to use as a backup for the original
            data. Default is "input.data.CUR_backup".
        """
        n_to_select_list = self._validate_n_to_select(
            n_to_select_list=n_to_select_list,
            n_total=self.n_frames,
            quantitiy="structure frames",
            file_out_list=file_out_list,
        )

        environments_r2 = self._create_global_environment()
        selected_indices = self._run_k1_CUR(
            environments_r2=environments_r2,
            n_to_select=max(n_to_select_list),
            weights=None,
        )
        dataset = Dataset(
            data_file=join(self.n2p2_directory, file_in),
        )
        dataset.write_data_file(file_out=join(self.n2p2_directory, file_backup))
        for i, n in enumerate(n_to_select_list):
            dataset.write_data_file(
                file_out=join(self.n2p2_directory, file_out_list[i]),
                conditions=(i in selected_indices[:n] for i in range(self.n_frames)),
            )
