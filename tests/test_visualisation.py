"""
Unit tests for `visualisation.py`
"""

import matplotlib.pyplot as plt
import pytest

from cc_hdnnp.visualisation import (
    plot_clustering,
    plot_data_histogram,
    plot_environments_histogram_2D,
    plot_epoch_histogram_1D,
    plot_epoch_histogram_2D,
    plot_epoch_scatter,
    plot_error_histogram,
    plot_lammps_temperature,
    plot_lammps_temperature_multiple,
    plot_learning_curve,
)


@pytest.fixture
def mock_plt_functions(mocker):
    """Mocks pyplot functions for tests"""
    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.subplot")
    mocker.patch("matplotlib.pyplot.scatter")
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.hist")
    mocker.patch("matplotlib.pyplot.hist2d")
    mocker.patch("matplotlib.pyplot.bar")
    mocker.patch("matplotlib.pyplot.xlabel")
    mocker.patch("matplotlib.pyplot.xticks")
    mocker.patch("matplotlib.pyplot.ylabel")
    mocker.patch("matplotlib.pyplot.legend")
    mocker.patch("matplotlib.pyplot.title")
    mocker.patch("matplotlib.pyplot.tight_layout")
    mocker.patch("matplotlib.pyplot.colorbar")


def test_plot_lammps_temperature(mock_plt_functions):
    """
    Test that (mocked) functions are called successfully.
    """
    plot_lammps_temperature(
        lammps_directory="tests/data/lammps", log_file="nve-t340.log"
    )
    plt.figure.assert_called_once()
    plt.plot.assert_called_once()
    plt.ylabel.assert_called_with("Temperature (K)")
    plt.title.assert_called_once_with("nve-t340.log")


def test_plot_lammps_temperature_multiple(mock_plt_functions):
    """
    Test that (mocked) functions are called successfully.
    """
    plot_lammps_temperature_multiple(
        lammps_directory="tests/data/lammps",
        log_file="{ensemble}-t{t}.log",
        temperatures=(340,),
    )
    plt.figure.assert_called_once()
    plt.subplot.assert_any_call(3, 1, 1)
    plt.subplot.assert_any_call(3, 1, 2)
    plt.subplot.assert_any_call(3, 1, 3)
    plt.plot.assert_called()
    plt.ylabel.assert_called_with("Temperature (K)")
    plt.title.assert_any_call("nve: 340K")
    plt.title.assert_any_call("nvt: 340K")
    plt.title.assert_any_call("npt: 340K")


def test_plot_learning_curve_error():
    """
    Test that an error is raised when unrecognised key is provided.
    """
    keys = ["unrecognised"]
    with pytest.raises(ValueError) as e:
        plot_learning_curve("tests/data/n2p2", keys=keys)

    assert str(e.value) == (
        "`key={}` not found in `learning-curve.out` headers: "
        "['epoch', 'RMSEpa_Etrain_pu', 'RMSEpa_Etest_pu', 'RMSE_Etrain_pu', 'RMSE_Etest_pu', "
        "'MAEpa_Etrain_pu', 'MAEpa_Etest_pu', 'MAE_Etrain_pu', 'MAE_Etest_pu', "
        "'RMSE_Ftrain_pu', 'RMSE_Ftest_pu', 'MAE_Ftrain_pu', 'MAE_Ftest_pu']"
        "".format(keys[0])
    )


def test_plot_learning_curve_default(mock_plt_functions):
    """
    Test that default arguments are chosen successfully.
    """
    plot_learning_curve("tests/data/n2p2")
    plt.figure.assert_called_once()
    plt.plot.assert_called()
    plt.xlabel.assert_called_with("Epoch")
    plt.legend.assert_called_once_with(["RMSEpa_Etrain_pu", "RMSEpa_Etest_pu"])


def test_plot_epoch_scatter(mock_plt_functions):
    """
    Test that plotting functions are called successfully.
    """
    plot_epoch_scatter("tests/data/n2p2", epoch=1)
    plt.figure.assert_called_once()
    plt.subplot.assert_any_call(2, 2, 1)
    plt.subplot.assert_any_call(2, 2, 2)
    plt.subplot.assert_any_call(2, 2, 3)
    plt.subplot.assert_any_call(2, 2, 4)
    plt.scatter.assert_called()
    plt.plot.assert_called()
    plt.xlabel.assert_called_with("Reference")
    plt.ylabel.assert_called_with("Network")
    plt.title.assert_any_call("Training Energies")
    plt.title.assert_any_call("Test Energies")
    plt.title.assert_any_call("Training Forces")
    plt.title.assert_any_call("Test Forces")


@pytest.mark.parametrize("energy_log, force_log", [(False, False), (True, True)])
def test_plot_epoch_histogram_2D(mock_plt_functions, energy_log: bool, force_log: bool):
    """
    Test that plotting functions are called successfully.
    """
    plot_epoch_histogram_2D(
        "tests/data/n2p2",
        epoch=1,
        energy_log=energy_log,
        force_log=force_log,
    )
    plt.figure.assert_called_once()
    plt.subplot.assert_any_call(2, 2, 1)
    plt.subplot.assert_any_call(2, 2, 2)
    plt.subplot.assert_any_call(2, 2, 3)
    plt.subplot.assert_any_call(2, 2, 4)
    plt.hist2d.assert_called()
    plt.xlabel.assert_called_with("Reference")
    plt.ylabel.assert_called_with("Network")
    plt.title.assert_any_call("Training Energies")
    plt.title.assert_any_call("Test Energies")
    plt.title.assert_any_call("Training Forces")
    plt.title.assert_any_call("Test Forces")


@pytest.mark.parametrize("combine_xyz", [False, True])
def test_plot_epoch_histogram_1D(mock_plt_functions, combine_xyz: bool):
    """
    Test that plotting functions are called successfully.
    """
    plot_epoch_histogram_1D(
        "tests/data/n2p2",
        epoch=1,
        combine_xyz=combine_xyz,
    )
    plt.figure.assert_called_once()
    plt.subplot.assert_called()
    plt.hist.assert_called()
    plt.xlabel.assert_any_call("Energy")
    if combine_xyz:
        plt.xlabel.assert_any_call("Forces")
    else:
        plt.xlabel.assert_any_call("Forces (x)")
        plt.xlabel.assert_any_call("Forces (y)")
        plt.xlabel.assert_any_call("Forces (z)")
    plt.title.assert_any_call("Network Training Energies")
    plt.title.assert_any_call("Reference Test Energies")
    plt.title.assert_any_call("Network Training Forces")
    plt.title.assert_any_call("Reference Test Forces")


@pytest.mark.parametrize("physical_units", [False, True])
@pytest.mark.parametrize("combine_xyz", [False, True])
def test_plot_error_histogram(
    mock_plt_functions, combine_xyz: bool, physical_units: bool
):
    """
    Test that plotting functions are called successfully.
    """
    plot_error_histogram(
        "tests/data/n2p2",
        epoch=1,
        combine_xyz=combine_xyz,
        physical_units=physical_units,
    )
    plt.figure.assert_called_once()
    plt.subplot.assert_called()
    plt.hist.assert_called()
    plt.xlabel.assert_any_call("Energy")
    if combine_xyz:
        plt.xlabel.assert_any_call("Forces")
    else:
        plt.xlabel.assert_any_call("Forces (x)")
        plt.xlabel.assert_any_call("Forces (y)")
        plt.xlabel.assert_any_call("Forces (z)")
    plt.title.assert_any_call("Training Energy Error")
    plt.title.assert_any_call("Test Energy Error")
    plt.title.assert_any_call("Training Forces Error")
    plt.title.assert_any_call("Test Forces Error")


@pytest.mark.parametrize("superimpose", [False, True])
def test_plot_data_histogram(mock_plt_functions, superimpose: bool):
    """
    Test that plotting functions are called successfully.
    """
    plot_data_histogram(
        data_files=("tests/data/n2p2/input.data", "tests/data/n2p2/input.data"),
        superimpose=superimpose,
    )
    plt.figure.assert_called_once()
    plt.subplot.assert_called()
    plt.hist.assert_called()
    plt.xlabel.assert_any_call("Energy")
    plt.xlabel.assert_any_call("Force")
    plt.title.assert_any_call("Reference Energies (tests/data/n2p2/input.data)")
    plt.title.assert_any_call("Reference Forces (tests/data/n2p2/input.data)")


def test_plot_clustering(mock_plt_functions):
    """
    Test that plotting functions are called successfully.
    """
    plot_clustering(
        elements=("C",),
        file_in="tests/data/n2p2/mocked_{}UR.out",
    )
    plt.figure.assert_called_once()
    plt.subplot.assert_called_once_with(1, 1, 1)
    plt.bar.assert_called()
    plt.xlabel.assert_called_once_with("Dataset")
    plt.ylabel.assert_called_once_with("Cluster labels")
    plt.title.assert_any_call("C")
    plt.legend.assert_called_once()


@pytest.mark.parametrize("log", [False, True])
def test_plot_environments_histogram_2D(mock_plt_functions, log: bool):
    """
    Test that plotting functions are called successfully.
    """
    plot_environments_histogram_2D(
        elements=("C",),
        file_environments="tests/data/n2p2/mocked_{}UR.out",
        log=log,
    )
    plt.figure.assert_called_once()
    plt.subplot.assert_called_once_with(1, 1, 1)
    plt.hist2d.assert_called()
    plt.xlabel.assert_called_once_with("Symmetry Function 1")
    plt.ylabel.assert_called_once_with("Symmetry Function 2")
    plt.title.assert_any_call("C")
