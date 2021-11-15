"""
Unit tests for `decomposer.py`
"""

from os import listdir, remove
from os.path import isfile
from shutil import copy, rmtree
from typing import List

from genericpath import isdir
import numpy as np
import pytest

from cc_hdnnp.data import Data
from cc_hdnnp.data_selection import Decomposer
from cc_hdnnp.structure import AllStructures, Species, Structure


@pytest.fixture
def data():
    species = Species(symbol="H", atomic_number=1, mass=1.0)
    structure = Structure(name="test", all_species=[species], delta_E=1.0, delta_F=1.0)

    yield Data(
        structures=AllStructures(structure),
        main_directory="tests/data",
        n2p2_bin="",
        lammps_executable="",
    )

    for file in listdir("tests/data/tests_output"):
        if isfile("tests/data/tests_output/" + file):
            remove("tests/data/tests_output/" + file)
        elif isdir("tests/data/tests_output/" + file):
            rmtree("tests/data/tests_output/" + file)


@pytest.mark.parametrize(
    "n_to_select_list, selection", [(None, ["#", "#", ""]), ([2], ["#", "", ""])]
)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_decompose_dataset_symf(
    data: Data,
    capsys: pytest.CaptureFixture,
    weight: bool,
    n_to_select_list: List[int],
    selection: List[str],
    verbosity: int,
):
    """
    Test that running CUR decomposition in "symf" mode results in the expected information
    being printed and written to file.
    """
    copy("tests/data/n2p2/input.nn", "tests/data/tests_output/input.nn")
    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    with open("tests/data/tests_output/input.nn", "a") as f:
        f.write(
            "symfunction_short H 2 H 1.0 1.0 1.0\n"
            "symfunction_short H 3 H H 1.0 1 1.0 1.0 1.0\n"
            "symfunction_short H 9 H H 1.0 1 1.0 1.0 1.0\n"
        )

    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    decomposer = Decomposer(data_controller=data, verbosity=verbosity)
    decomposer.run_CUR_symf(weight=weight, n_to_select_list=n_to_select_list)

    assert list(decomposer._atom_environments.keys()) == ["H"]
    assert np.all(
        decomposer._atom_environments["H"]
        == np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
                [[0.6, 1.2, 1.8], [0.7, 1.4, 2.1]],
                [[0.9, 1.8, 2.7], [1.0, 2.0, 3.0]],
            ],
            dtype="float32",
        )
    )

    text = capsys.readouterr().out
    if verbosity <= 0:
        assert text == ""
    else:
        assert "Values read from file in " in text
        assert "Element: H\n" in text
        assert "Selected indices in " in text
        if n_to_select_list is None:
            assert "Selecting 1 out of 3 symmetry functions\n" in text
        else:
            assert "Selecting 1 out of 3 symmetry functions\n" not in text
        if verbosity >= 2:
            assert "Step   1: selected      2 with a score of " in text
        else:
            assert "Step   1: selected      2 with a score of " not in text

    file = "tests/data/tests_output/input.nn"
    assert isfile(file + ".CUR_backup")
    with open(file + ".CUR_backup") as f:
        lines = f.readlines()
        assert lines[-3] == "symfunction_short H 2 H 1.0 1.0 1.0\n"
        assert lines[-2] == "symfunction_short H 3 H H 1.0 1 1.0 1.0 1.0\n"
        assert lines[-1] == "symfunction_short H 9 H H 1.0 1 1.0 1.0 1.0\n"

    assert isfile(file)
    with open(file) as f:
        lines = f.readlines()
        assert lines[-3] == selection[-3] + "symfunction_short H 2 H 1.0 1.0 1.0\n"
        assert (
            lines[-2] == selection[-2] + "symfunction_short H 3 H H 1.0 1 1.0 1.0 1.0\n"
        )
        assert (
            lines[-1] == selection[-1] + "symfunction_short H 9 H H 1.0 1 1.0 1.0 1.0\n"
        )


def test_decompose_dataset_data(
    data: Data,
    capsys: pytest.CaptureFixture,
):
    """
    Test that running CUR decomposition in "data" mode results in the expected information
    being printed and written to file.
    """
    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")

    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    decomposer = Decomposer(data_controller=data)
    decomposer.run_CUR_data()

    assert list(decomposer._atom_environments.keys()) == ["H"]
    assert np.all(
        decomposer._atom_environments["H"]
        == np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
                [[0.6, 1.2, 1.8], [0.7, 1.4, 2.1]],
                [[0.9, 1.8, 2.7], [1.0, 2.0, 3.0]],
            ],
            dtype="float32",
        )
    )

    text = capsys.readouterr().out
    assert "Values read from file in " in text
    assert "Selecting 1 out of 3 structure frames\n" in text
    assert "Selected indices in " in text
    assert "[2]\n" in text

    file = "tests/data/tests_output/input.data"
    assert isfile(file + ".CUR_backup")
    with open(file + ".CUR_backup") as f:
        lines = f.readlines()
        assert len(lines) == 33

    assert isfile(file)
    with open(file) as f:
        lines = f.readlines()
        assert len(lines) == 11
        assert lines[-3] == "energy       2.00000000\n"


@pytest.mark.parametrize(
    "n_to_select_list, error",
    [
        (
            [101],
            "All entries in `n_to_select_list` must be less than the number of quantity, "
            + "but they were [101], 100",
        ),
        ([0], "All entries in `n_to_select_list` must be at least 1, but were [0]"),
        (
            [1, 2],
            "`n_to_select_list` and `file_out_list` must have the same lengths, but were 2, 1",
        ),
    ],
)
def test_validate_n_to_select(
    data: Data,
    n_to_select_list: List[int],
    error: str,
):
    """
    Test ValueErrors are raised if `n_to_select_list` is greater than `n_total` or less than 0.
    """
    decomposer = Decomposer(data_controller=data)
    with pytest.raises(ValueError) as e:
        decomposer._validate_n_to_select(
            n_to_select_list=n_to_select_list,
            n_total=100,
            quantitiy="quantity",
            file_out_list=[""],
        )

    assert str(e.value) == error


def test_calculate_symf_weights_unrecognised(
    data: Data,
):
    """
    Test ValueError raised if symmetry functions are unrecognised.
    """
    file_data = "tests/data/tests_output/input.data"
    file_in = "tests/data/tests_output/input.nn"
    file_weights = "tests/data/tests_output/symf_weights.log"
    error = "Weighting for symfunction type 0 is not supported."

    copy("tests/data/n2p2/input.nn", "tests/data/tests_output/input.nn")
    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    with open("tests/data/tests_output/input.nn", "a") as f:
        f.write("symfunction_short H 0 H 1.0 1.0 1.0\n")

    decomposer = Decomposer(data_controller=data)
    with pytest.raises(ValueError) as e:
        decomposer._calculate_symf_weights(
            file_data=file_data,
            file_in=file_in,
            file_weights=file_weights,
        )

    assert str(e.value) == error


@pytest.mark.parametrize(
    "text, error",
    [
        (
            "symfunction_short H 2 H   1.0   1.0 1.0\n",
            "Should have 3 weights for element H, but had 1",
        ),
        (
            "symfunction_short H 2 H   1.0   1.0 inf\n"
            + "symfunction_short H 3 H H 1.0 1 1.0 1.0 1.0\n"
            + "symfunction_short H 9 H H 1.0 1 1.0 1.0 1.0\n",
            "Should have finite weights for element H, but some were not",
        ),
        (
            "symfunction_short H 2 H   1.0   1.0 0.0\n"
            + "symfunction_short H 3 H H 1.0 1 1.0 1.0 1.0\n"
            + "symfunction_short H 9 H H 1.0 1 1.0 1.0 1.0\n",
            "All weights should be positive for element H, but some were not",
        ),
    ],
)
def test_calculate_symf_weights_errors(
    data: Data,
    text: str,
    error: str,
):
    """
    Test ValueErrors if symmetry functions are unrecognised or give non-finit or non-positive
    values.
    """
    file_data = "tests/data/tests_output/input.data"
    file_in = "tests/data/tests_output/input.nn"
    file_weights = "tests/data/tests_output/symf_weights.log"

    copy("tests/data/n2p2/input.nn", "tests/data/tests_output/input.nn")
    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    with open("tests/data/tests_output/input.nn", "a") as f:
        f.write(text)

    decomposer = Decomposer(data_controller=data)
    with pytest.raises(ValueError) as e:
        decomposer._calculate_symf_weights(
            file_data=file_data,
            file_in=file_in,
            file_weights=file_weights,
        )

    assert str(e.value) == error


def test_n_frames_error(
    data: Data,
):
    """
    Test ValueError raised if atom environments have different numbers of frames.
    """
    error = "Not all elements have the same number of frames (1, 0)"

    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")

    decomposer = Decomposer(data_controller=data)
    decomposer._atom_environments = {"H": np.zeros(1), "He": np.zeros(0)}
    with pytest.raises(ValueError) as e:
        decomposer.n_frames

    assert str(e.value) == error


def test_run_k1_CUR_error(
    data: Data,
):
    """
    Test ValueError raised if there's a non-finite entry in `environments_r2`.
    """
    error = "Non-finite entries {} in array at position {}".format(
        [np.inf], np.array([[0], [0]])
    )

    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")

    decomposer = Decomposer(data_controller=data)
    with pytest.raises(ValueError) as e:
        decomposer._run_k1_CUR(
            environments_r2=np.array([[np.inf]]), n_to_select=1, weights=None
        )

    assert str(e.value) == error
