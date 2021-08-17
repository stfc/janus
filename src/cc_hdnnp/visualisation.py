"""
Utility functions for reading from file and plotting the performance of the network.
"""

from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt


def plot_learning_curve(n2p2_directory: str, keys: List[str] = None):
    """
    Reads "learning-curve.out" from the specified directory and plots the specified quantities.

    Parameters
    ----------
    n2p2_directory : str
        Directory to find "learning-curve.out" in.
    keys : list of str
        The keys of the data to plot. Possible values are:
          - "RMSEpa_Etrain_pu" RMSE of training energies per atom (physical units)
          - "RMSEpa_Etest_pu"  RMSE of test energies per atom (physical units)
          - "RMSE_Etrain_pu"   RMSE of training energies (physical units)
          - "RMSE_Etest_pu"    RMSE of test energies (physical units)
          - "MAEpa_Etrain_pu"  MAE of training energies per atom (physical units)
          - "MAEpa_Etest_pu"   MAE of test energies per atom (physical units)
          - "MAE_Etrain_pu"    MAE of training energies (physical units)
          - "MAE_Etest_pu"     MAE of test energies (physical units)
          - "RMSE_Ftrain_pu"   RMSE of training forces (physical units)
          - "RMSE_Ftest_pu"    RMSE of test forces (physical units)
          - "MAE_Ftrain_pu"    MAE of training forces (physical units)
          - "MAE_Ftest_pu"     MAE of test forces (physical units)
        Default is `None`, in which case "RMSEpa_Etrain_pu" and "RMSEpa_Etest_pu" will be
        plotted.
    """
    content = []
    epochs = []
    if keys is None:
        keys = ["RMSEpa_Etrain_pu", "RMSEpa_Etest_pu"]

    with open(join(n2p2_directory, "learning-curve.out")) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("#    epoch"):
            # Header for the rest of the table
            headers = line[1:].split()
        elif not line.startswith("#"):
            # Table content
            content.append(line.split())
            epochs.append(int(line.split()[0]))

    plt.figure(figsize=(12, 6))
    for key in keys:
        try:
            index = headers.index(key)
        except ValueError as e:
            raise ValueError(
                "`key={0}` not found in `learning-curve.out` headers: "
                "{1}".format(key, headers)
            ) from e
        values = [float(row[index]) for row in content]
        plt.plot(epochs[1:], values[1:])
        plt.xticks(range(epochs[1], epochs[-1] + 1))
        plt.xlabel("Epoch")
    plt.legend(keys)


def _read_epoch_file(file: str, index: int = 1) -> Tuple[List[float]]:
    """
    Read `file` and return the reference and network predicted values.
    `index` should be `1` for energies and `2` for force files.

    Parameters
    ----------
    file : str
        File to read.
    index : int, optional
        Corresponds to the position of the reference value in `file`. Default is `1`.

    Returns
    -------
    tuple of list of float
        The first entry is the list of reference values, the second is the list of network
        predictions.
    """
    reference_values = []
    nnp_values = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith("#"):
            reference_values.append(float(line.split()[index]))
            nnp_values.append(float(line.split()[index + 1]))

    return reference_values, nnp_values


def plot_epoch(n2p2_directory: str, epoch: int, subsample_forces: int = 512):
    """
    For `n2p2_directory`, load the files corresponding to `epoch` and scatter plot the
    reference values against the network predictions. Due to the large number of force values
    (3 * number of atoms * number of structures), can subsample these to cut down on the time
    taken to plot.

    Parameters
    ----------
    n2p2_directory : str
        Directory to find training/testing files in.
    epoch : int
        The epoch to plot data for
    subsample_forces : int, optional
        Only selects every nth value for the forces. Default is 512.

    """

    energy_train_file = join(n2p2_directory, "trainpoints.{:06d}.out").format(
        epoch
    )
    energy_test_file = join(n2p2_directory, "testpoints.{:06d}.out").format(epoch)
    force_train_file = join(n2p2_directory, "trainforces.{:06d}.out").format(
        epoch
    )
    force_test_file = join(n2p2_directory, "testforces.{:06d}.out").format(epoch)

    plt.figure(figsize=(12, 12))
    plt.tight_layout()

    energy_train_ref, energy_train_nn = _read_epoch_file(energy_train_file)
    energy_test_ref, energy_test_nn = _read_epoch_file(energy_test_file)
    force_train_ref, force_train_nn = _read_epoch_file(force_train_file, index=2)
    force_test_ref, force_test_nn = _read_epoch_file(force_test_file, index=2)
    # Subsample as we have ~ 1M forces
    force_train_ref = force_train_ref[::subsample_forces]
    force_train_nn = force_train_nn[::subsample_forces]
    force_test_ref = force_test_ref[::subsample_forces]
    force_test_nn = force_test_nn[::subsample_forces]

    energy_guide = [
        min(
            min(energy_train_ref),
            min(energy_train_nn),
            min(energy_test_ref),
            min(energy_test_nn)
        ),
        max(
            max(energy_train_ref),
            max(energy_train_nn),
            max(energy_test_ref),
            max(energy_test_nn)
        ),
    ]
    force_guide = [
        min(min(force_train_ref), min(force_train_nn), min(force_test_ref), min(force_test_nn)),
        max(max(force_train_ref), max(force_train_nn), max(force_test_ref), max(force_test_nn)),
    ]

    plt.subplot(2, 2, 1)
    plt.scatter(energy_train_ref, energy_train_nn, s=1)
    plt.plot(energy_guide, energy_guide, "k--")
    plt.xlim()
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Training Energies")

    plt.subplot(2, 2, 2)
    plt.scatter(energy_test_ref, energy_test_nn, s=1, c="tab:orange")
    plt.plot(energy_guide, energy_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Test Energies")

    plt.subplot(2, 2, 3)
    plt.scatter(force_train_ref, force_train_nn, s=1)
    plt.plot(force_guide, force_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Training Forces")

    plt.subplot(2, 2, 4)
    plt.scatter(force_test_ref, force_test_nn, s=1, c="tab:orange")
    plt.plot(force_guide, force_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Test Forces")
