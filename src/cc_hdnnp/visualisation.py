"""
Utility functions for reading from file and plotting the performance of the network.
"""

from os.path import exists, join
from typing import List, Tuple

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from cc_hdnnp.file_operations import read_data_file, read_normalisation


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
        tick_step = (epochs[-1] + 1) // 30 + 1
        plt.xticks(range(epochs[0], epochs[-1] + 1, tick_step))
        plt.xlabel("Epoch")
    plt.legend(keys)


def _read_epoch_file(file: str, index: int = 1) -> Tuple[np.ndarray]:
    """
    Read `file` and return the reference and network predicted values.
    `index` should be `1` for energies and `2` for force files.

    Parameters
    ----------
    file : str
        File to read.
    index : int, optional
        Corresponds to the position of the reference value in `file`. Default is `1`.

    TODO update retrun type
    Returns
    -------
    tuple of ndarray
        The first entry is the array of reference values, the second is the array of network
        predictions.
    """
    if not exists(file):
        print("{} not found, return `0.`".format(file))
        return np.array([0.0]), np.array([0.0])

    reference_values = []
    nnp_values = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith("#"):
            reference_values.append(float(line.split()[index]))
            nnp_values.append(float(line.split()[index + 1]))

    return np.array(reference_values), np.array(nnp_values)


def plot_epoch_scatter(n2p2_directory: str, epoch: int, subsample_forces: int = 512):
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

    energy_train_file = join(n2p2_directory, "trainpoints.{:06d}.out").format(epoch)
    energy_test_file = join(n2p2_directory, "testpoints.{:06d}.out").format(epoch)
    force_train_file = join(n2p2_directory, "trainforces.{:06d}.out").format(epoch)
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
            min(energy_test_nn),
        ),
        max(
            max(energy_train_ref),
            max(energy_train_nn),
            max(energy_test_ref),
            max(energy_test_nn),
        ),
    ]
    force_guide = [
        min(
            min(force_train_ref),
            min(force_train_nn),
            min(force_test_ref),
            min(force_test_nn),
        ),
        max(
            max(force_train_ref),
            max(force_train_nn),
            max(force_test_ref),
            max(force_test_nn),
        ),
    ]

    plt.subplot(2, 2, 1)
    plt.scatter(energy_train_ref, energy_train_nn, s=1)
    plt.plot(energy_guide, energy_guide, "k--")
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


def plot_epoch_histogram_2D(n2p2_directory: str, epoch: int, bins: int = 10):
    """
    For `n2p2_directory`, load the files corresponding to `epoch` and histogrsm plot the
    reference values against the network predictions.

    Parameters
    ----------
    n2p2_directory : str
        Directory to find training/testing files in.
    epoch : int
        The epoch to plot data for
    bins : int, optional
        The number of bins along each axis. Default is 10.

    """

    energy_train_file = join(n2p2_directory, "trainpoints.{:06d}.out").format(epoch)
    energy_test_file = join(n2p2_directory, "testpoints.{:06d}.out").format(epoch)
    force_train_file = join(n2p2_directory, "trainforces.{:06d}.out").format(epoch)
    force_test_file = join(n2p2_directory, "testforces.{:06d}.out").format(epoch)

    plt.figure(figsize=(12, 12))
    plt.tight_layout()

    energy_train_ref, energy_train_nn = _read_epoch_file(energy_train_file)
    energy_test_ref, energy_test_nn = _read_epoch_file(energy_test_file)
    force_train_ref, force_train_nn = _read_epoch_file(force_train_file, index=2)
    force_test_ref, force_test_nn = _read_epoch_file(force_test_file, index=2)

    energy_guide = [
        min(
            min(energy_train_ref),
            min(energy_train_nn),
            min(energy_test_ref),
            min(energy_test_nn),
        ),
        max(
            max(energy_train_ref),
            max(energy_train_nn),
            max(energy_test_ref),
            max(energy_test_nn),
        ),
    ]
    force_guide = [
        min(
            min(force_train_ref),
            min(force_train_nn),
            min(force_test_ref),
            min(force_test_nn),
        ),
        max(
            max(force_train_ref),
            max(force_train_nn),
            max(force_test_ref),
            max(force_test_nn),
        ),
    ]

    plt.subplot(2, 2, 1)
    plt.hist2d(energy_train_ref, energy_train_nn, bins=bins, cmap="viridis")
    plt.plot(energy_guide, energy_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Training Energies")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.hist2d(energy_test_ref, energy_test_nn, bins=bins, cmap="plasma")
    plt.plot(energy_guide, energy_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Test Energies")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.hist2d(
        force_train_ref, force_train_nn, bins=bins, cmap="viridis", norm=LogNorm()
    )
    plt.plot(force_guide, force_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Training Forces")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.hist2d(force_test_ref, force_test_nn, bins=bins, cmap="plasma", norm=LogNorm())
    plt.plot(force_guide, force_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Test Forces")
    plt.colorbar()


def plot_epoch_histogram_1D(
    n2p2_directory: str, epoch: int, bins: int = 10, combine_xyz: bool = True
):
    """
    For `n2p2_directory`, load the files corresponding to `epoch` and plot a histogram of
    reference values and network predictions.

    Parameters
    ----------
    n2p2_directory : str
        Directory to find training/testing files in.
    epoch : int
        The epoch to plot data for
    bins : int, optional
        The number of bins to use in the histogram. Default is 10.
    combine_xyz : bool, optional
        If `True`, plot all xyz components of forces on the same histogram. Default is `True`.
    """
    energy_train_file = join(n2p2_directory, "trainpoints.{:06d}.out").format(epoch)
    energy_test_file = join(n2p2_directory, "testpoints.{:06d}.out").format(epoch)
    force_train_file = join(n2p2_directory, "trainforces.{:06d}.out").format(epoch)
    force_test_file = join(n2p2_directory, "testforces.{:06d}.out").format(epoch)

    energy_train_ref, energy_train_nn = _read_epoch_file(energy_train_file)
    energy_test_ref, energy_test_nn = _read_epoch_file(energy_test_file)
    force_train_ref, force_train_nn = _read_epoch_file(force_train_file, index=2)
    force_test_ref, force_test_nn = _read_epoch_file(force_test_file, index=2)

    if combine_xyz:
        n_rows = 4
    else:
        n_rows = 8

    plt.figure(figsize=(12, 6 * n_rows))
    plt.tight_layout()

    plt.subplot(n_rows, 2, 1)
    plt.hist(energy_train_ref, bins=bins)
    plt.xlabel("Energy")
    plt.title("Reference Training Energies")

    plt.subplot(n_rows, 2, 2)
    plt.hist(energy_test_ref, bins=bins, color="tab:orange")
    plt.xlabel("Energy")
    plt.title("Reference Test Energies")

    plt.subplot(n_rows, 2, 3)
    plt.hist(energy_train_nn, bins=bins)
    plt.xlabel("Energy")
    plt.title("Network Training Energies")

    plt.subplot(n_rows, 2, 4)
    plt.hist(energy_test_nn, bins=bins, color="tab:orange")
    plt.xlabel("Energy")
    plt.title("Network Test Energies")

    if combine_xyz:
        plt.subplot(n_rows, 2, 5)
        plt.hist(force_train_ref, bins=bins)
        plt.xlabel("Forces")
        plt.title("Reference Training Forces")

        plt.subplot(n_rows, 2, 6)
        plt.hist(force_test_ref, bins=bins, color="tab:orange")
        plt.xlabel("Forces")
        plt.title("Reference Test Forces")

        plt.subplot(n_rows, 2, 7)
        plt.hist(force_train_nn, bins=bins)
        plt.xlabel("Forces")
        plt.title("Network Training Forces")

        plt.subplot(n_rows, 2, 8)
        plt.hist(force_test_nn, bins=bins, color="tab:orange")
        plt.xlabel("Forces")
        plt.title("Network Test Forces")
    else:
        plt.subplot(n_rows, 2, 5)
        plt.hist(force_train_ref[::3], bins=bins)
        plt.xlabel("Forces (x)")
        plt.title("Reference Training Forces")

        plt.subplot(n_rows, 2, 6)
        plt.hist(force_test_ref[::3], bins=bins, color="tab:orange")
        plt.xlabel("Forces (x)")
        plt.title("Reference Test Forces")

        plt.subplot(n_rows, 2, 7)
        plt.hist(force_train_nn[::3], bins=bins)
        plt.xlabel("Forces (x)")
        plt.title("Network Training Forces")

        plt.subplot(n_rows, 2, 8)
        plt.hist(force_test_nn[::3], bins=bins, color="tab:orange")
        plt.xlabel("Forces (x)")
        plt.title("Network Test Forces")

        plt.subplot(n_rows, 2, 9)
        plt.hist(force_train_ref[1::3], bins=bins)
        plt.xlabel("Forces (y)")
        plt.title("Reference Training Forces")

        plt.subplot(n_rows, 2, 10)
        plt.hist(force_test_ref[1::3], bins=bins, color="tab:orange")
        plt.xlabel("Forces (y)")
        plt.title("Reference Test Forces")

        plt.subplot(n_rows, 2, 11)
        plt.hist(force_train_nn[1::3], bins=bins)
        plt.xlabel("Forces (y)")
        plt.title("Network Training Forces")

        plt.subplot(n_rows, 2, 12)
        plt.hist(force_test_nn[1::3], bins=bins, color="tab:orange")
        plt.xlabel("Forces (y)")
        plt.title("Network Test Forces")

        plt.subplot(n_rows, 2, 13)
        plt.hist(force_train_ref[2::3], bins=bins)
        plt.xlabel("Forces (z)")
        plt.title("Reference Training Forces")

        plt.subplot(n_rows, 2, 14)
        plt.hist(force_test_ref[2::3], bins=bins, color="tab:orange")
        plt.xlabel("Forces (z)")
        plt.title("Reference Test Forces")

        plt.subplot(n_rows, 2, 15)
        plt.hist(force_train_nn[2::3], bins=bins)
        plt.xlabel("Forces (z)")
        plt.title("Network Training Forces")

        plt.subplot(n_rows, 2, 16)
        plt.hist(force_test_nn[2::3], bins=bins, color="tab:orange")
        plt.xlabel("Forces (z)")
        plt.title("Network Test Forces")


def plot_error_histogram(
    n2p2_directory: str,
    epoch: int,
    bins: int = 10,
    combine_xyz: bool = True,
    energy_scale: str = "linear",
    force_scale: str = "linear",
    physical_units: bool = False,
):
    """
    TODO
    """
    energy_train_file = join(n2p2_directory, "trainpoints.{:06d}.out").format(epoch)
    energy_test_file = join(n2p2_directory, "testpoints.{:06d}.out").format(epoch)
    force_train_file = join(n2p2_directory, "trainforces.{:06d}.out").format(epoch)
    force_test_file = join(n2p2_directory, "testforces.{:06d}.out").format(epoch)

    energy_train_ref, energy_train_nn = _read_epoch_file(energy_train_file)
    energy_test_ref, energy_test_nn = _read_epoch_file(energy_test_file)
    force_train_ref, force_train_nn = _read_epoch_file(force_train_file, index=2)
    force_test_ref, force_test_nn = _read_epoch_file(force_test_file, index=2)

    if physical_units:
        conv_energy, conv_length = read_normalisation(join(n2p2_directory, "input.nn"))
        energy_train_ref /= conv_energy
        energy_train_nn /= conv_energy
        energy_test_ref /= conv_energy
        energy_test_nn /= conv_energy
        force_train_ref /= conv_energy / conv_length
        force_train_nn /= conv_energy / conv_length
        force_test_ref /= conv_energy / conv_length
        force_test_nn /= conv_energy / conv_length

    if combine_xyz:
        n_rows = 2
    else:
        n_rows = 4

    plt.figure(figsize=(12, 6 * n_rows))
    plt.tight_layout()

    plt.subplot(n_rows, 2, 1)
    plt.hist(energy_train_nn - energy_train_ref, bins=bins)
    plt.xlabel("Energy")
    plt.yscale(energy_scale)
    plt.title("Training Energy Error")

    plt.subplot(n_rows, 2, 2)
    plt.hist(energy_test_nn - energy_test_ref, bins=bins, color="tab:orange")
    plt.xlabel("Energy")
    plt.yscale(energy_scale)
    plt.title("Test Energy Error")

    if combine_xyz:
        plt.subplot(n_rows, 2, 3)
        plt.hist(force_train_nn - force_train_ref, bins=bins)
        plt.xlabel("Forces")
        plt.yscale(force_scale)
        plt.title("Training Forces Error")

        plt.subplot(n_rows, 2, 4)
        plt.hist(force_test_nn - force_test_ref, bins=bins, color="tab:orange")
        plt.xlabel("Forces")
        plt.yscale(force_scale)
        plt.title("Test Forces Error")

    else:
        plt.subplot(n_rows, 2, 3)
        plt.hist(force_train_nn[::3] - force_train_ref[::3], bins=bins)
        plt.yscale(force_scale)
        plt.xlabel("Forces (x)")
        plt.title("Training Forces Error")

        plt.subplot(n_rows, 2, 4)
        plt.hist(
            force_test_nn[::3] - force_test_ref[::3], bins=bins, color="tab:orange"
        )
        plt.yscale(force_scale)
        plt.xlabel("Forces (x)")
        plt.title("Test Forces Error")

        plt.subplot(n_rows, 2, 5)
        plt.hist(force_train_nn[1::3] - force_train_ref[1::3], bins=bins)
        plt.yscale(force_scale)
        plt.xlabel("Forces (y)")
        plt.title("Training Forces Error")

        plt.subplot(n_rows, 2, 6)
        plt.hist(
            force_test_nn[1::3] - force_test_ref[1::3], bins=bins, color="tab:orange"
        )
        plt.yscale(force_scale)
        plt.xlabel("Forces (y)")
        plt.title("Test Forces Error")

        plt.subplot(n_rows, 2, 7)
        plt.hist(force_train_nn[2::3] - force_train_ref[2::3], bins=bins)
        plt.yscale(force_scale)
        plt.xlabel("Forces (z)")
        plt.title("Training Forces Error")

        plt.subplot(n_rows, 2, 8)
        plt.hist(
            force_test_nn[2::3] - force_test_ref[2::3], bins=bins, color="tab:orange"
        )
        plt.yscale(force_scale)
        plt.xlabel("Forces (z)")
        plt.title("Test Forces Error")


def plot_data_histogram(
    data_files: List[str],
    bins: int = 10,
    energy_scale: str = "linear",
    force_scale: str = "linear",
    superimpose: bool = True,
    alpha=1.0,
):
    """
    TODO update docstring for new functionality
    """
    n_rows = 1 if superimpose else len(data_files)
    energies = []
    forces = []
    for i, data_file in enumerate(data_files):
        data = read_data_file(data_file)
        energy = []
        force = []
        for frame in data:
            energy.append(frame[4])
            for atomic_force in frame[3]:
                for j in (0, 1, 2):
                    force.append(atomic_force[j])
        if i == 0:
            energy_min = min(energy)
            energy_max = max(energy)
            force_min = min(force)
            force_max = max(force)
        else:
            energy_min = min(energy_min, min(energy))
            energy_max = max(energy_max, max(energy))
            force_min = min(force_min, min(force))
            force_max = max(force_max, max(force))
        energies.append(energy)
        forces.append(force)

    plt.figure(figsize=(12, 6 * n_rows))
    for i, data_file in enumerate(data_files):
        if superimpose:
            energy_position = 1
            force_position = 2
        else:
            energy_position = 2 * i + 1
            force_position = 2 * i + 2

        plt.subplot(n_rows, 2, energy_position)
        plt.hist(energies[i], bins=bins, range=(energy_min, energy_max), alpha=alpha)
        plt.xlabel("Energy")
        plt.yscale(energy_scale)
        plt.title("Reference Energies ({})".format(data_file))
        plt.legend(data_files)

        plt.subplot(n_rows, 2, force_position)
        plt.hist(forces[i], bins=bins, range=(force_min, force_max), alpha=alpha)
        plt.xlabel("Force")
        plt.yscale(force_scale)
        plt.title("Reference Forces ({})".format(data_file))
        plt.legend(data_files)
