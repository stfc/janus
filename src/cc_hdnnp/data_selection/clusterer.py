"""
Class and functions for clustering data.
"""

from os.path import join
import time
from typing import Tuple

import numpy as np
from numpy.lib.function_base import iterable
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN


from .dataselector import DataSelector


class Clusterer(DataSelector):
    """
    Performs clustering of atomic environments based on evaluated symmetry functions saved to
    "atomic-env.G".

    Parameters
    ----------
    atoms_per_frame: int,
        The number of atoms present in each frame, needed to read the atomic environment data
        from file.
    data_controller: Data,
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

    def _cluster_environments(
        self,
        file_out: str,
        cluster_estimator: BaseEstimator,
        environments: np.ndarray,
        output_shape: Tuple[
            int,
        ] = None,
    ):
        """Cluster the `environments`, print the labels and number of samples for each label
        and write all the labels to `file_out`.

        Parameters
        ----------
        file_out: str
            The complete file_path to write the labels to.
        cluster_estimator: BaseEstimator
            The algorithm to use for clustering, with parameters set.
        environments: np.ndarray
            Array of float with shape (N, M) where N is the number of samples and M the number
            of features.
        output_shape: tuple of int, optional
            If defined, will reshape the labels before writing them to file. Default is None.
        """
        t1 = time.time()

        db = cluster_estimator.fit(environments)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if self.verbosity >= 1:
            print("{} labels assigned".format(n_clusters))
            print(
                "Noise    : {:6d} {:3.3f}%".format(n_noise, 100 * n_noise / len(labels))
            )
            for i in range(n_clusters):
                print(
                    "Label {:3d}: {:6d} {:3.3f}%".format(
                        i, np.sum(labels == i), 100 * np.sum(labels == i) / len(labels)
                    )
                )

        if output_shape:
            labels = labels.reshape(output_shape)

        with open(file_out, "w") as f:
            for frame_labels in labels:
                if iterable(frame_labels):
                    frame_labels.sort()
                    [f.write("{:3d}".format(label)) for label in frame_labels]
                    f.write("\n")
                else:
                    f.write("{:3d}\n".format(frame_labels))

        if self.verbosity >= 1:
            t2 = time.time()
            print("Clustered in {} s".format(t2 - t1))

    def run_atom_clustering(
        self,
        file_out: str = "clustered_{}.data",
        cluster_estimator: BaseEstimator = None,
    ):
        """
        Performs clustering of individual atoms, by element,
        based on `self.atomic environments`.

        Parameters
        ----------
        file_out: str = "clustered_{}.data"
            Formatable file name to write output to, relative to `self.n2p2_directory`.
            Will be formatted with either a chemical symbol.
        cluster_estimator: BaseEstimator = None
            The SciKitLearn clustering algorithm to use. Default is None, in which case DBSCAN
            is used with default parameters.
        """
        if cluster_estimator is None:
            cluster_estimator = DBSCAN()
            if self.verbosity >= 1:
                print("Defaulting to DBSCAN for clustering")

        for i, element in enumerate(self.elements):
            if self.verbosity >= 1:
                print("\nElement: {}".format(element))

            environments = self.get_environments(element=element)
            environments_r2 = environments.reshape(
                (self.n_frames * self.n_atoms_list[i], self.n_symf_list[i])
            )
            self._cluster_environments(
                file_out=join(self.n2p2_directory, file_out.format(element)),
                cluster_estimator=cluster_estimator,
                environments=environments_r2,
                output_shape=(self.n_frames, self.n_atoms_list[i]),
            )

    def run_frame_clustering(
        self,
        file_out: str = "clustered_frames.data",
        cluster_estimator: BaseEstimator = None,
    ):
        """
        Performs clustering based of frames based on `self.atomic_environments`.

        Parameters
        ----------
        file_out: str = "clustered_frame.data"
            File name to write output to, relative to `self.n2p2_directory`.
        cluster_estimator: BaseEstimator = None
            The SciKitLearn clustering algorithm to use. Default is None, in which case DBSCAN
            is used with default parameters.
        """
        if cluster_estimator is None:
            cluster_estimator = DBSCAN()
            if self.verbosity >= 1:
                print("Defaulting to DBSCAN for clustering")

        all_environments = np.zeros((self.n_frames, 0))
        for i, element in enumerate(self.elements):
            all_environments = np.append(
                all_environments,
                self.get_environments(element).reshape(
                    (self.n_frames, self.n_atoms_list[i] * self.n_symf_list[i])
                ),
                axis=-1,
            )
        self._cluster_environments(
            file_out=join(self.n2p2_directory, file_out),
            cluster_estimator=cluster_estimator,
            environments=all_environments,
        )
