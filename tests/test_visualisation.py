"""
Unit tests for `visualisation.py`
"""

import matplotlib.pyplot as plt
import pytest

from cc_hdnnp.visualisation import plot_epoch_scatter, plot_learning_curve


@pytest.fixture
def mock_plt_functions(mocker):
    """Mocks pyplot functions for tests"""
    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.subplot")
    mocker.patch("matplotlib.pyplot.scatter")
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.xlabel")
    mocker.patch("matplotlib.pyplot.xticks")
    mocker.patch("matplotlib.pyplot.ylabel")
    mocker.patch("matplotlib.pyplot.legend")
    mocker.patch("matplotlib.pyplot.title")
    mocker.patch("matplotlib.pyplot.tight_layout")


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
