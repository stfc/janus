"""
Class and functions for selecting data based on separation.
"""

from os.path import join
import time
from typing import Callable, Dict, List, Literal, Union

import numpy as np

from janus.dataset import Dataset
from .dataselector import DataSelector
from ..file_readers import read_nn_settings, read_scaling


class Separator(DataSelector):
    """
    Selects frames from the of atomic environments based on their separation in terms of the
    evaluated symmetry functions saved to "atomic-env.G".

    Parameters
    ----------
    data_controller: Controller
        Controller object used to get the n2p2 directory to work from, and elements present.
    n2p2_directory_index: int = 0,
        Used in conjunction with `data_controller` to get the n2p2 directory to work from.
        Default is 0.
    verbosity: int = 1,
        How much information to print during operation. 0 will result in no printing,
        1 prints summaries. Default is 1.
    dtype: str = "float32",
        Data type to use in the numpy arrays. Default is "float32".
    """

    def _validate_metric_function(
        self, criteria: Union[Literal["mean"], float]
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Checks that `criteria` is a valid str or float, and if so returns a function
        for calculating the metric for comparing frames.

        Parameters
        ----------
        criteria: Union[Literal["mean"], float]
            Indicates to criteria to compare the separation of points in terms of their
            symmetry functions, either as a str or a float. The latter is interpreted as
            a quantile, for example 0.5 will use the median separation.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], np.ndarray]
            Function to calculate the metric for each proposed frame from the proposed
            and comparison frames.
        """
        if isinstance(criteria, float):
            if criteria > 1 or criteria < 0:
                raise ValueError(
                    "`criteria` must be between 0 and 1, but was {}".format(criteria)
                )

            def f(
                environments_compare: np.ndarray, environments_proposed: np.ndarray
            ) -> np.ndarray:
                rmsd = (
                    np.mean(
                        ((environments_compare - environments_proposed)) ** 2,
                        axis=4,
                    )
                    ** 0.5
                )
                return np.quantile(rmsd, q=criteria, axis=(0, 2, 3))

        elif criteria == "mean":

            def f(
                environments_compare: np.ndarray, environments_proposed: np.ndarray
            ) -> np.ndarray:
                return (
                    np.mean(
                        ((environments_compare - environments_proposed)) ** 2,
                        axis=(0, 2, 3, 4),
                    )
                    ** 0.5
                )

        else:
            raise ValueError(
                "`criteria` must be a quantile (float) between 0 and 1 "
                "or 'mean', but was {}".format(criteria)
            )

        return f

    def _select_close_frames(
        self, environments: np.ndarray, value: Union[float, np.ndarray]
    ) -> List[int]:
        """
        Parameters
        ----------
            environments: np.ndarray
                Array of float with shape (N, M, L) where N is the number of frames,
                M the number of atoms and L the number of symmetry functions.
            value: Union[float, np.ndarray]
                Either a single float or an array with length L to compare each
                symmetry function against.

        Returns
        -------
        List[int]
            List of frame indices with length L. Each is the first frame in the dataset
            which has a symmetry function that evaluates close to `value`.
        """
        # Any atom (axis=1) that is close to `value` will return True for the frame it's in
        # np.argmax over the frames will give the first index of a frame containing one of
        # those atoms (preventing having muliple frames for the same value)
        # Cast to list so we can easily append to other indices
        return list(
            np.argmax(np.any(np.isclose(environments, value, atol=0), axis=1), axis=0)
        )

    def _select_min_max_scaled_frames(
        self, scaling_settings: Dict[str, str]
    ) -> List[int]:
        """
        Selects frames that have a symmetry function which evaluates to either min or max
        value used for the scaling process.

        Parameters
        ----------
        scaling_settings: Dict[str, str]
            The settings used to scale the data, which must contain "scale_min_short" and
            "scale_max_short" as keys.

        Returns
        -------
        List[int]
            List of frame indices. Each is the first frame in the dataset which has a symmetry
            function that evaluates close to either the min or max value they were scaled to.
        """
        indicies = []
        for element in self.elements:
            indicies += self._select_close_frames(
                environments=self.get_environments(element),
                value=float(scaling_settings["scale_min_short"]),
            )
            indicies += self._select_close_frames(
                environments=self.get_environments(element),
                value=float(scaling_settings["scale_max_short"]),
            )
        return indicies

    def _select_sigma_scaled_frames(
        self, scaling: Dict[str, Dict[str, List[float]]]
    ) -> List[int]:
        """
        Selects frames that have a symmetry function which evaluates to either min or max
        value present in the dataset after accounting for sigma scaling.

        Parameters
        ----------
        scaling: Dict[str, Dict[str, List[float]]]
            The results of scaling performed on the dataset. Outer dictionary has keys
            corresponding to the chemical symbol of each element, the inner dictionary
            must have the keys "min", "max", "mean", "sigma" with a list of float as
            value.

        Returns
        -------
        List[int]
            List of frame indices. Each is the first frame in the dataset which has a symmetry
            function that evaluates close to either the min or max value after sigma scaling.
        """
        indicies = []
        for element in self.elements:
            min_array = np.array(scaling[element]["min"])
            max_array = np.array(scaling[element]["max"])
            mean_array = np.array(scaling[element]["mean"])
            sigma_array = np.array(scaling[element]["sigma"])
            indicies += self._select_close_frames(
                environments=self.get_environments(element),
                value=(min_array - mean_array) / sigma_array,
            )
            indicies += self._select_close_frames(
                environments=self.get_environments(element),
                value=(max_array - mean_array) / sigma_array,
            )
        return indicies

    def _select_unscaled_frames(
        self, scaling: Dict[str, Dict[str, List[float]]]
    ) -> List[int]:
        """
        Selects frames that have a symmetry function which evaluates to either min or max
        value present in the dataset when no scaling was applied.

        Parameters
        ----------
        scaling: Dict[str, Dict[str, List[float]]]
            The results of scaling performed on the dataset. Outer dictionary has keys
            corresponding to the chemical symbol of each element, the inner dictionary
            must have the keys "min", "max", "mean", "sigma" with a list of float as
            value.

        Returns
        -------
        List[int]
            List of frame indices. Each is the first frame in the dataset which has a symmetry
            function that evaluates close to either the min or max value.
        """
        indicies = []
        for element in self.elements:
            min_array = np.array(scaling[element]["min"])
            max_array = np.array(scaling[element]["max"])
            indicies += self._select_close_frames(
                environments=self.get_environments(element),
                value=min_array,
            )
            indicies += self._select_close_frames(
                environments=self.get_environments(element),
                value=max_array,
            )
        return indicies

    def _select_extreme_frames(self) -> List[int]:
        """
        Returns
        -------
        List[int]
            List of frame indices that contain extremal evaluations of the symmetry functions.
        """
        scaling_settings = read_nn_settings(
            join(self.n2p2_directory, "input.nn"),
            requested_settings=[
                "scale_symmetry_functions",
                "scale_symmetry_functions_sigma",
                "scale_min_short",
                "scale_max_short",
                "center_symmetry_functions",
            ],
        )
        scaling = read_scaling(
            join(self.n2p2_directory, "scaling.data"),
            self.elements,
        )
        # Calculate the min and max values for symmetry functions taking scaling into
        # account. Frames that contain a minimal or maximal value are automatically
        # selected.
        if (
            "scale_symmetry_functions" in scaling_settings
            and "scale_symmetry_functions_sigma" in scaling_settings
        ):
            raise ValueError(
                "Both scale_symmetry_functions and scale_symmetry_functions_sigma "
                "were present in settings file."
            )
        elif "scale_symmetry_functions" in scaling_settings:
            if (
                "scale_min_short" not in scaling_settings
                or "scale_max_short" not in scaling_settings
            ):
                raise ValueError(
                    "If scale_symmetry_functions is set, both scale_min_short and "
                    "scale_max_short must be present."
                )
            return self._select_min_max_scaled_frames(scaling_settings=scaling_settings)
        elif "scale_symmetry_functions_sigma" in scaling_settings:
            return self._select_sigma_scaled_frames(scaling=scaling)
        else:
            return self._select_unscaled_frames(scaling=scaling)

    def run_separation_selection(
        self,
        n_frames_to_select: int,
        n_frames_to_propose: int,
        n_frames_to_compare: int,
        select_extreme_frames: bool = True,
        starting_frame_indices: List[int] = None,
        criteria: Union[Literal["mean"], float] = "mean",
        seed: int = None,
        file_in: str = "input.data",
        file_out: str = "input.data",
        file_backup: str = "input.data.rebuild_backup",
    ):
        """
        Taking the frames in `file_in` as the original dataset, reconstructs a new,
        smaller dataset and writes it to `file_out`.

        The selection of new structures is based on the separation of their symmetry functions,
        or fingerprints. A number of frames are proposed, and the euclidean distance of their
        fingerprint vectors to the vectors of already accepted frames are compared. Note that
        as multiple atomic environments are present in each frame, this difference is taken
        between all atoms of the same species. Using `criteria`, the proposed frame(s) that
        are most different from already accepted frames are added to the new dataset.

        Because of the high dimensionality involved, this is done in batches, with the number of
        frames to propose, compare against and select being configurable.

        Parameters
        ----------
        n_frames_to_select: int
            The number of frames to select from each batch.
        n_frames_to_propose: int
            The number of frames proposed at each batch.
        n_frames_to_compare: int
            The number of already accepted frames to compare against at each batch.
        select_extreme_frames: bool, optional
            If True, then frames containing a maximal or minimal symmetry function value is
            automatically selected and used in addition to `starting_frame_indices`. Note that
            if multiple frames minimise the same function, only one of these will be added.
            Default is True
        starting_frame_indices: list of int, optional
            If provided, these frames will be used as the initial set to compare against.
            If `None`, and `select_extreme_frames` is False, then `n_frames_compare` will be
            randomly selected instead. Default is `None`.
        criteria: float or "mean", optional
            If a float between 0 and 1, defines the quantile to take when comparing frames.
            For example, 1 would mean the maximum separation between two atomic environments
            is used to determine which frames to add, 0.5 would take the median separation.
            If "mean", then the mean of all environments is compared. Default is "mean".
        seed: int, optional
            The seed is used to randomly order the frames for selection. Default is `None`.
        dtype: str, optional
            numpy data type to use. Default is "float32".
        n2p2_directory_index: str, optional
            The index of the directory within `self.n2p2_directories` containing the weights
            files. Default is 0.
        file_in: str, optional
            File path of the n2p2 structure file, relative to `n2p2_directory`,
            to read from. Default is "input.data".
        file_out: str, optional
            File path of the n2p2 structure file, relative to `n2p2_directory`,
            to write to. Default is "input.data".
        file_backup: str, optional
            File path of the n2p2 structure file, relative to `n2p2_directory`, to copy
            the original `file_in` to. Default is "input.data.minimum_separation_backup".
        """
        metric_function = self._validate_metric_function(criteria=criteria)

        np.random.seed(seed)
        remove_indices = np.array([])
        frame_indices = np.random.permutation(self.n_frames)

        if starting_frame_indices is None:
            starting_frame_indices = []

        print(starting_frame_indices)
        if select_extreme_frames:
            starting_frame_indices += self._select_extreme_frames()
        print(starting_frame_indices)

        # If starting_frame_indices provided, use those. Otherwise select starting frames
        # at random from the shuffled list of all frames.
        if len(starting_frame_indices) == 0:
            selected_indices = frame_indices[:n_frames_to_compare]
            frame_indices = frame_indices[n_frames_to_compare:]
        else:
            selected_indices = np.unique(starting_frame_indices)
            for starting_index in selected_indices:
                frame_indices = frame_indices[frame_indices != starting_index]

        print()

        if self.verbosity >= 1:
            print(
                "Starting separation selection with the following frames selected:\n{}\n"
                "".format(selected_indices)
            )

        while len(frame_indices) > 0:
            t1 = time.time()
            frames_compare = selected_indices[-n_frames_to_compare:]
            frames_proposed = frame_indices[:n_frames_to_propose]
            frame_indices = frame_indices[n_frames_to_propose:]
            total_metric = np.zeros(len(frames_proposed), dtype=self.dtype)

            for element in self.elements:
                environments = self.get_environments(element=element)[frames_compare]
                environments_compare = environments.reshape(
                    environments.shape[0],
                    1,
                    environments.shape[1],
                    1,
                    environments.shape[2],
                )
                environments = self.get_environments(element=element)[frames_proposed]
                environments_proposed = environments.reshape(
                    1,
                    environments.shape[0],
                    1,
                    environments.shape[1],
                    environments.shape[2],
                )

                total_metric += metric_function(
                    environments_compare, environments_proposed
                )

            max_separation_indicies = frames_proposed[
                np.argsort(total_metric)[-n_frames_to_select:]
            ]
            min_separation_indicies = frames_proposed[
                np.argsort(total_metric)[:-n_frames_to_select]
            ]
            selected_indices = np.concatenate(
                (selected_indices, max_separation_indicies)
            )
            remove_indices = np.concatenate((remove_indices, min_separation_indicies))

            if self.verbosity >= 2:
                print(
                    "Proposed indices:\n{0}\n"
                    "Difference metric summed over all elements:\n{1}\n".format(
                        frames_proposed, total_metric
                    )
                )
                print("Selected indices:\n{}\n".format(max_separation_indicies))
                print("Time taken: {}\n".format(time.time() - t1))

        dataset = Dataset(
            data_file=join(self.n2p2_directory, file_in),
        )
        dataset.write_data_file(file_out=join(self.n2p2_directory, file_backup))
        dataset.write_data_file(
            file_out=join(self.n2p2_directory, file_out),
            conditions=(i in selected_indices for i in range(self.n_frames)),
        )

        return selected_indices
