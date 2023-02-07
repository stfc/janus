"""
Utility functions for reading from file and plotting the performance of the network.
"""

from os.path import join
from typing import Iterable, List, Tuple, Union

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pickle

from .dataset import Dataset
from .file_readers import read_lammps_log, read_nn_settings

def plot_lammps_data(
    lammps_directory: str,
    log_file: str,
    timesteps_range: Tuple[int, int] = (0, None),
    output_headers= ["Temp"]
):
    """
    Plots temperatures from a single LAMMPS log file.

    Parameters
    ----------
    lammps_directory: str
        The directory containing `log_file`.
    log_file: str
        The filepath of the LAMMPS log file, relative to `lammps_directory`.
    timesteps_range: Tuple[int, int] = None
        Sets the upper and lower limit on the timesteps to plot.
        Optional, default is `(0, None)` which plots all timesteps.
    """
    _, _, _, dataset = read_lammps_log(
        dump_lammpstrj=1,
        log_lammps_file=join(lammps_directory, log_file),
        output_headers = output_headers
    )
    for key in dataset.keys():
        ds = dataset[key]
        plt.figure(figsize=(12, 6))
        plt.plot(ds[timesteps_range[0] : timesteps_range[1]])
        plt.ylabel(key)
        plt.title(log_file)
        plt.show()


def plot_lammps_temperature(
    lammps_directory: str,
    log_file: str,
    timesteps_range: Tuple[int, int] = (0, None),
):
    """
    Plots temperatures from a single LAMMPS log file.

    Parameters
    ----------
    lammps_directory: str
        The directory containing `log_file`.
    log_file: str
        The filepath of the LAMMPS log file, relative to `lammps_directory`.
    timesteps_range: Tuple[int, int] = None
        Sets the upper and lower limit on the timesteps to plot.
        Optional, default is `(0, None)` which plots all timesteps.
    """
    _, _, _, dataset = read_lammps_log(
        dump_lammpstrj=1,
        log_lammps_file=join(lammps_directory, log_file),
    )
    temperatures = dataset["Temp"]
    plt.figure(figsize=(12, 6))
    plt.plot(temperatures[timesteps_range[0] : timesteps_range[1]])
    plt.ylabel("Temperature (k)")
    plt.title(log_file)
    plt.show()


def plot_lammps_temperature_multiple(
    lammps_directory: str,
    log_file: str = "{ensemble}-t{t}.log",
    ensembles: Iterable[str] = ("nve", "nvt", "npt"),
    temperatures: Iterable[int] = (300,),
    timesteps_range: Tuple[int, int] = (0, None),
):
    """
    Plots temperatures from several LAMMPS log files that match the provided arguments.

    Parameters
    ----------
    lammps_directory: str
        The directory containing `log_file`.
    log_file: str = "{ensemble}-t{t}.log"
        The filepath of a LAMMPS log file, relative to `lammps_directory` that can be
        formatted with `ensemble` and `t`.
    ensembles: Iterable[str] = ("nve", "nvt", "npt")
        An iterable sequence of str representing ensembles. Will attempt to format
        `log_file` with each in turn, and plot the temperatures from that file.
    temperatures: Iterable[int] = (300,)
        An iterable sequence of int representing temperatures. Will attempt to format
        `log_file` with each in turn, and plot the temperatures from that file.
    timesteps_range: Tuple[int, int] = None
        Sets the upper and lower limit on the timesteps to plot.
        Optional, default is `(0, None)` which plots all timesteps.
    """
    plt.figure(figsize=(12, 6 * len(ensembles) * len(temperatures)))
    i = 1
    for ensemble in ensembles:
        for t in temperatures:
            plt.subplot(len(ensembles) * len(temperatures), 1, i)
            _, _, _, lammps_dataset = read_lammps_log(
                dump_lammpstrj=1,
                log_lammps_file=join(
                    lammps_directory, log_file.format(ensemble=ensemble, t=t)
                ),
            )
            lammps_temperatures = lammps_dataset["Temp"]
            plt.plot(lammps_temperatures[timesteps_range[0] : timesteps_range[1]])
            plt.ylabel("Temperature (K)")
            plt.title(f"{ensemble}: {t}K")
            i += 1


def plot_learning_curve(
    n2p2_directory: str, keys: List[str] = None, epoch_range: List[int] = None
):
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
    epoch_range: List[int] = None
        Defines the epochs to plot. Default is None, in which case all epochs will be plotted.
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
            epoch = int(line.split()[0])
            if epoch_range is None or epoch in epoch_range:
                content.append(line.split())
                epochs.append(epoch)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    for key in keys:
        try:
            index = headers.index(key)
        except ValueError as e:
            raise ValueError(
                "`key={0}` not found in `learning-curve.out` headers: "
                "{1}".format(key, headers)
            ) from e
        values = [float(row[index]) for row in content]
        plt.plot(epochs, values)
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

    Returns
    -------
    tuple of ndarray
        The first entry is the array of reference values, the second is the array of network
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


def plot_epoch_histogram_2D(
    n2p2_directory: str,
    epoch: int,
    bins: int = 10,
    force_log: bool = True,
    energy_log: bool = True,
):
    """
    For `n2p2_directory`, load the files corresponding to `epoch` and histogram plot the
    reference values against the network predictions.

    Parameters
    ----------
    n2p2_directory : str
        Directory to find training/testing files in.
    epoch : int
        The epoch to plot data for
    bins : int, optional
        The number of bins along each axis. Default is 10.
    force_log: bool = True
        Whether to use a log scale when plotting the forces. Optional, default is True.
    energy_log:bool = True
        Whether to use a log scale when plotting the energy. Optional, default is True.
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
    if energy_log:
        plt.hist2d(
            energy_train_ref, energy_train_nn, bins=bins, cmap="viridis", norm=LogNorm()
        )
    else:
        plt.hist2d(energy_train_ref, energy_train_nn, bins=bins, cmap="viridis")
    plt.plot(energy_guide, energy_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Training Energies")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    if energy_log:
        plt.hist2d(
            energy_test_ref, energy_test_nn, bins=bins, cmap="plasma", norm=LogNorm()
        )
    else:
        plt.hist2d(energy_test_ref, energy_test_nn, bins=bins, cmap="plasma")
    plt.plot(energy_guide, energy_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Test Energies")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    if force_log:
        plt.hist2d(
            force_train_ref, force_train_nn, bins=bins, cmap="viridis", norm=LogNorm()
        )
    else:
        plt.hist2d(force_train_ref, force_train_nn, bins=bins, cmap="viridis")
    plt.plot(force_guide, force_guide, "k--")
    plt.xlabel("Reference")
    plt.ylabel("Network")
    plt.title("Training Forces")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    if force_log:
        plt.hist2d(
            force_test_ref, force_test_nn, bins=bins, cmap="plasma", norm=LogNorm()
        )
    else:
        plt.hist2d(force_test_ref, force_test_nn, bins=bins, cmap="plasma")
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
    Plots the difference between the predicted and reference energies and forces as
    histograms.

    Parameters
    ----------
    n2p2_directory: str
        The directory containing the files written during training.
    epoch: int
        The epoch from the training to plot.
    bins: int = 10
        The number of bins to use when plotting the histograms. Optional, default is `10`.
    combine_xyz: bool = True
        If `True`, plot all xyz components of forces on the same histogram. Default is `True`.
    energy_scale: str = "linear"
        The scale to use for the energy values, should be "linear" or "log".
        Optional, default is "linear".
    force_scale: str = "linear"
        The scale to use for the force values, should be "linear" or "log".
        Optional, default is "linear".
    physical_units: bool = False
        Whether to use the headers of the "input.nn" file to convert normalised values back
        into physical units. If the data is not normalised, this should be `False`.
        Optional, default is `False`.
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
        settings = read_nn_settings(
            settings_file=join(n2p2_directory, "input.nn"),
            requested_settings=("conv_energy", "conv_length"),
        )
        conv_energy = float(settings["conv_energy"])
        conv_length = float(settings["conv_length"])
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
    alpha: float = 1.0,
    density: bool = False,
    ec: str = None,
    histtype: str = "bar",
):
    """
    Plots the energy and force values present in a series of reference datasets. This can be
    useful for identifying the presence of extreme values which may impede training.

    Parameters
    ----------
    data_files: List[str]
        A list of filepaths to n2p2 "input.data" files.
    bins: int = 10
        The number of bins to use when plotting the histograms. Optional, default is `10`.
    energy_scale: str = "linear"
        The scale to use for the energy values, should be "linear" or "log".
        Optional, default is "linear".
    force_scale: str = "linear"
        The scale to use for the force values, should be "linear" or "log".
        Optional, default is "linear".
    superimpose: bool = True
        Whether to plot the histograms for subsequent entries in `data_files` on the same axes
        or not. If `True`, then the first entry in `data_files` should contain all points to be
        plotted, with subsequent entries being subsets to ensure full visibility.
        Optional, default is `True`.
    alpha: float = 1.0
        Sets the transparency of the plotted bars. `1.0` is opaque. Optional, default is `1.0`.
    density: bool = False
        Whether to plot the probability density. Optional, default is `False`.
    ec: str = None
        Edge colour of histogram. Optional, default is `None`.
    histtype: str = "bar"
        Type of histogram to draw. Optional, default is "bar".
    """
    n_rows = 1 if superimpose else len(data_files)
    energies = []
    forces = []
    for i, data_file in enumerate(data_files):
        data = Dataset(data_file=data_file)
        energy = data.all_energies
        force = np.concatenate(data.all_forces, axis=0).flatten()
        if i == 0:
            energy_min = min(energy)
            energy_max = max(energy)
            force_min = np.amin(force)
            force_max = np.amax(force)
        else:
            energy_min = min(energy_min, min(energy))
            energy_max = max(energy_max, max(energy))
            force_min = min(force_min, np.amin(force))
            force_max = max(force_max, np.amax(force))
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
        plt.hist(energies[i], bins=bins, range=(energy_min, energy_max), alpha=alpha, density=density, ec=ec, histtype=histtype)
        plt.xlabel("Energy")
        plt.yscale(energy_scale)
        plt.title("Reference Energies ({})".format(data_file))
        plt.legend(data_files)

        plt.subplot(n_rows, 2, force_position)
        plt.hist(forces[i], bins=bins, range=(force_min, force_max), alpha=alpha, density=density, ec=ec, histtype=histtype)
        plt.xlabel("Force")
        plt.yscale(force_scale)
        plt.title("Reference Forces ({})".format(data_file))
        plt.legend(data_files)


def plot_clustering(elements: Iterable[str], file_in: str = "clustered_{}.data"):
    """
    Plot the results of clustering as vertical bars. Each element in `elements` is shown on
    a separate subplot. Each bar in the plot represents a line the input file, and if it has
    multiple labels on the line then the bar's colour is split between these labels.

    Parameters
    ----------
    elements: Iterable[str]
        The elements to plot for, if the data was split by elements. If not, then `["all"]`
        should be used.
    file_in: str = "clustered_{}.data"
        The complete file path to read from. Will be formatted with the entries in `elements`.
    """
    ncols = len(elements)
    plt.figure(figsize=(6 * ncols, 6), constrained_layout=True)
    for i, element in enumerate(elements):
        data = []
        with open(file_in.format(element)) as f:
            line = f.readline()
            while line:
                data.append([float(value) for value in line.split()])
                line = f.readline()
        data = np.array(data, dtype=int)
        labels = np.unique(data)
        bottom = np.zeros(len(data))
        plt.subplot(1, ncols, i + 1)
        for j in labels:
            label_counts = np.sum(data == j, axis=-1)
            legend_label = "Noise" if j == -1 else str(j)
            plt.bar(
                range(len(data)),
                label_counts,
                width=1,
                bottom=bottom,
                label=legend_label,
            )
            bottom += label_counts
        plt.tick_params(
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.xlabel("Dataset")
        plt.ylabel("Cluster labels")
        plt.title(element)
        plt.legend()


def plot_environments_histogram_2D(
    elements: Iterable[str], file_environments: str, bins: int = 10, log: bool = False
):
    """
    Plots histograms for atomic environments in terms of two symmetry functions selected by CUR
    for each element.

    Parameters
    ----------
    elements: Iterable[str]
        A list of chemical symbols, which is used to format `file_environments`.
        A plot will be made for each element, using the two best symmetry functions
        for that element to come out of the CUR decomposition.
    file_environments: str
        The filepath to the files containing the evaluated environments in terms of the
        first two symmetry functions.
    bins: int = 10
        The number of bins along each axis. Default is 10.
    log: bool = False
        Whether to plot the log of frequency.
    """
    n = len(elements)
    plt.figure(figsize=(12, 12 * n))
    plt.tight_layout()

    for i, element in enumerate(elements):
        with open(file_environments.format(element)) as f:
            x = np.array(f.readline().split(), dtype=float)
            y = np.array(f.readline().split(), dtype=float)
        plt.subplot(n, 1, i + 1)
        if log:
            plt.hist2d(x, y, bins=bins, cmap="viridis", norm=LogNorm())
        else:
            plt.hist2d(x, y, bins=bins, cmap="viridis")
        plt.xlabel("Symmetry Function 1")
        plt.ylabel("Symmetry Function 2")
        plt.title(element)
        plt.colorbar()


def _read_predict_file(file: str) -> Tuple[np.ndarray]:
    """
    Return the reference and network predicted values from nnp-dataset stored in `file`.

    Parameters
    ----------
    file : str
        File containing energy or force outputs from nnp-dataset to read.
    Returns
    -------
    tuple of ndarray
        The first entry is the array of reference values, the second is the array of network
        predictions.
    """
    with open(file, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if not line.startswith("#"):
                start_idx = i
                break
        lines = lines[start_idx:]
        ref = np.zeros(len(lines))
        n2p2 = np.zeros(len(lines))
        for i, line in enumerate(lines):
            ref[i] = float(line.split()[2])
            n2p2[i] = float(line.split()[3])
    return ref, n2p2


def plot_dataset_predictions(
    n2p2_directory: str,
    bins: int = 50,
    density: bool = True,
    range: List[float] = None,
    alpha: float = 0.5,
    ec: str = 'k',
    histtype: str = "stepfilled",
    quantities: dict = {"energy": "Ha"},
    plot_type: str = "raw",
):
    """
    For `n2p2_directory`, load the files corresponding to energies or forces predicted by
    nnp-dataset and histogram plot the reference values against the network predictions.

    Parameters
    ----------
    n2p2_directory: str
        Directory containing energy.comp and forces.comp created by nnp-dataset.
    bins: int = 50
        The number of energy or force bins. Optional, default is 50.
    density: bool = True
        Whether to plot the probability density. Optional, default is `True`.
    range: List[float] = None
        Histogram x-axis range. Optional, default is `None`.
    alpha: float = 0.5
        Sets the transparency of the plotted bars. `1.0` is opaque. Optional, default is `0.5`.
    ec: str = 'k'
        Edge colour of histogram. Optional, default is `k`.
    histtype: str = "stepfilled"
        Type of histogram to draw. Optional, default is "stepfilled".
    """
    for quantity, unit in quantities.items():

        if quantity.lower() == "energy":
            comp_file = "energy.comp"
        elif quantity.lower() == "force":
            comp_file = "forces.comp"
        else:
            raise ValueError(
                f"{quantity} is not a valid quantity. Options: 'energy', 'force'"
        )
        ref, n2p2 = _read_predict_file(f'{n2p2_directory}/{comp_file}')
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size': 14})
        label = None

        if plot_type == "raw":
            plt.hist(
                ref,
                bins=bins,
                label='Reference',
                density=density,
                range=range,
                alpha=alpha,
                ec=ec,
                histtype=histtype,
            )
            data = n2p2
            label = "Prediction"
            xlabel = f'{quantity} / {unit}'

        elif plot_type == "diff":
            data = np.abs(ref - n2p2)
            symbol = quantity.upper()[0]
            xlabel= "$|" + symbol + "_{ref} - " + symbol + "_{n2p2}|$"

        elif plot_type == "percentage_diff":
            data = np.where(ref==0, np.nan, 100 * (ref - n2p2) / ref)
            symbol = quantity.upper()[0]
            xlabel = "$" + symbol + "_{n2p2}$ error / %"

        else:
            raise ValueError(
                f"{plot_type} is not a valid plot type. "
                "Options: 'raw', 'diff', 'percentage_diff"
        )

        plt.hist(
            data,
            bins=bins,
            label=label,
            density=density,
            range=range,
            alpha=alpha,
            ec=ec,
            histtype=histtype,
        )
        if plot_type == "raw":
            plt.legend()
        plt.xlabel(xlabel)
        if density:
            plt.ylabel('Density')
        else:
            plt.ylabel('Count')
        plt.show()


def plot_all_RDFs_with_errors(
    file_in: str,
    units: str = "nm",
    dir_out: str = None,
):
    """
    Plots reference and predicted RDFs, and the associated errors
    for each species pair saved in a .pkl file.

    Parameters
    ----------
    file_in: str
        The filepath of the .pkl files containing RDF data to be plotted.
    units: str = "nm"
        Units of distance being plotted. Default is `nm`.
    dir_out: str = None
        Directory to save plots. Default is `None`.
    """
    with open(file_in, 'rb') as f:
        data = pickle.load(f)

    file_out = None

    for name, ref in data['ref_rdf'].items():
        if dir_out is not None:
            file_out = f"{dir_out}/{name}_RDF.pdf"

        plot_RDF_with_error(
            ref,
            data['test_rdf'][name],
            data['rdf_errors'][name],
            name,
            units=units,
            file_out=file_out,
        )


def _truncate_data(
    ref: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncates predicted data to match the length of reference data.

    Parameters
    ----------
    ref: Tuple
        Reference RDF data.
    test: Tuple
        Predicted RDF data.

    Returns
    -------
    tuple of ndarray
        The first entry is the array of truncated distances, and
        the second is the array of truncated prediction data.
    """
    if test[0].max() > ref[0].max():
        mask_num = np.sum(test[0] < ref[0].max()) + 1
        dist_masked = test[0][:mask_num]
        test_masked = test[1][:mask_num]
    else:
        dist_masked = test[0][:]
        test_masked = test[1][:]
    return dist_masked, test_masked


def plot_RDF_with_error(
    ref: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    error: List[Union[np.ndarray, np.float64]],
    name: str,
    units: str = "nm",
    file_out: str = None,
):
    """
    Plots the reference and predicted RDF, and the associated error,
    for a pair of species. Based on functions in aml.score.rdf

    Parameters
    ----------
    ref: Tuple
        Reference RDF data.
    test: Tuple
        Predicted RDF data.
    error: Tuple
        Mean absolute error data between the reference and predicted RDFs.
    name: str
        Name of species pair RDF data refers to.
    units: str = "nm"
        Units of distance being plotted. Default is `nm`.
    file_out: str = None
        File to save plots. Default is `None`.
    """
    # Plot settings
    cm2in = 1/2.54
    fig = plt.figure(
        figsize=(20*cm2in, 20*cm2in),
        constrained_layout=True,
    )
    plt.rcParams.update({'font.size': 14})
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[2., 1.])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    dist_masked, test_masked = _truncate_data(ref, test)

    # Plot reference and predicted RDF
    ax0.plot(ref[0], ref[1], color='black',
                label="Reference RDF", lw=2)
    ax0.plot(dist_masked, test_masked, color='red', dashes=(0.5, 1.5), dash_capstyle='round',
                label="Test RDF", lw=2)

    # Plot error
    ax1.plot(error[0], error[1], color='black', lw=2)

    # Formatting
    ax0.set_ylabel("RDF")
    ax1.set_ylabel("Absolute Error")
    ax1.set_xlabel(f"Distance / {units}")
    ax0.set_title(f"Species: {name}")
    ax0.set_xticklabels([])
    plt.show()

    if file_out is not None:
        plt.savefig(
            fname=file_out,
            format="pdf",
            dpi=300,
        )


def plot_all_RDFs_combined(
    files_in: List[str],
    labels: List[str] = None,
    units: str = "nm",
    file_out: str = None,
):
    """
    Plots comparisons of RDFs saved in different .pkl files for all species pairs.

    Parameters
    ----------
    files_in: List[str]
        The filepath of the .pkl files containing predicted RDF data to be plotted.
    labels: List[str]
        Labels for plot legend, with the reference label first,
        followed by the labels for the files listed in `files_in`. Default is `None`.
    units: str = "nm"
        Units of distance being plotted. Default is `nm`.
    file_out: str = None
        File to save plots. Default is `None`.
    """
    label = None
    #Plot reference from first file
    with open(files_in[0], 'rb') as f:
                data = pickle.load(f)

    for name, ref in data['ref_rdf'].items():
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size': 14})

        # Plot reference data
        if labels is not None:
            label = labels[0]
        plt.plot(
            ref[0],
            ref[1],
            color='black',
            label=label,
            lw=1,
        )

        for i, file_in in enumerate(files_in):
            with open(file_in, 'rb') as f:
                data = pickle.load(f)
                try:
                    test = data['test_rdf'][name]
                    error = data['rdf_errors'][name]
                except KeyError:
                    swap_name = name.rsplit('-')[1] + "-" + name.rsplit('-')[0]
                    test = data['test_rdf'][swap_name]
                    error = data['rdf_errors'][swap_name]

            dist_masked, test_masked = _truncate_data(ref, test)

            if labels is not None:
                label = labels[i+1]
            plt.plot(
                dist_masked,
                test_masked,
                linestyle="--",
                lw=1.5,
                label=label,
            )

        # Formatting
        plt.ylabel("RDF")
        plt.xlabel(f"Distance / {units}")
        plt.title(f"{name}")
        if labels is not None:
            plt.legend()

        if file_out is not None:
            plt.savefig(
                fname=f"{file_out}_{name}.pdf",
                format="pdf",
                dpi=300,
                bbox_inches = "tight",
            )
        plt.show()


def plot_average_RDFs(
    files_in: List[str],
    labels: List[str] = None,
    units: str = "nm",
    file_out: str = None,
):
    """
    Plots comparisons of the average RDFs across all atom pairs
    saved in different .pkl files.

    Parameters
    ----------
    files_in: List[str]
        The filepath of the .pkl files containing predicted RDF data to be plotted.
    labels: List[str]
        Labels for plot legend, with the reference label first,
        followed by the labels for the files listed in `files_in`. Default is `None`.
    units: str = "nm"
        Units of distance being plotted. Default is `nm`.
    file_out: str = None
        File to save plots. Default is `None`.
    """
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    with open(files_in[0], 'rb') as f:
            data = pickle.load(f)

    # Plot reference data
    ref_sum = None
    for name, ref in data['ref_rdf'].items():
        if ref_sum is None:
            ref_sum = np.zeros(len(ref[1]))
        ref_sum = np.add(ref_sum, ref[1])

    if labels is not None:
        label = labels[0]
    plt.plot(
        ref[0],
        ref_sum,
        color='black',
        label=label,
        lw=1,
    )

    # Plot prediction data
    for i, file_in in enumerate(files_in):

        with open(file_in, 'rb') as f:
            data = pickle.load(f)

        test_data_sum = None
        for name, ref in data['ref_rdf'].items():
            try:
                test = data['test_rdf'][name]
            except KeyError:
                swap_name = name.rsplit('-')[1] + "-" + name.rsplit('-')[0]
                test = data['test_rdf'][swap_name]

            dist_masked, test_masked = _truncate_data(ref, test)

            if test_data_sum is None:
                test_data_sum = np.zeros(len(test_masked))
            test_data_sum = np.add(test_data_sum, test_masked)

        if labels is not None:
            label = labels[i+1]
        plt.plot(
            dist_masked,
            test_data_sum,
            linestyle="--",
            lw=1.5,
            label=label,
        )

    # Formatting
    plt.ylabel("RDF")
    plt.xlabel(f"Distance / {units}")
    if labels is not None:
        plt.legend()

    if file_out is not None:
        plt.savefig(
            fname=f"{file_out}_sum.pdf",
            format="pdf",
            dpi=300,
            bbox_inches = "tight",
        )
    plt.show()


def print_RDF_accuracies(file_in: str):
    """
    Prints a table of RDF accuracies from a .pkl file.

    Parameters
    ----------
    file_in: List[str]
        The filepath of the .pkl file containing predicted and reference RDF data.
    """
    with open(file_in, 'rb') as f:
        data = pickle.load(f)
    accuracies = []
    print("Label  | Accuracy / %")
    print("==========================")
    for name, ref_data in data['ref_rdf'].items():
        error = data['rdf_errors'][name]
        accuracy = float(100 * (1 - np.average(error[2])))
        accuracies.append(accuracy)
        print(f"{name.ljust(7)}| {accuracy}")
    print(f"Mean   | {np.average(accuracies)}")
    print("==========================")
