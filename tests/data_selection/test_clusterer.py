"""
Unit tests for `clusterer.py`
"""

from os import listdir, remove
from os.path import isfile
from shutil import copy, rmtree

from genericpath import isdir
import numpy as np
import pytest

from cc_hdnnp.data import Data
from cc_hdnnp.data_selection import Clusterer
from cc_hdnnp.structure import AllSpecies, AllStructures, Species, Structure


@pytest.fixture
def data():
    species = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species = AllSpecies(species)
    structure = Structure(
        name="test", all_species=all_species, delta_E=1.0, delta_F=1.0
    )

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


@pytest.mark.parametrize("verbosity", [True, False])
def test_run_atom_clustering(
    data: Data,
    capsys: pytest.CaptureFixture,
    verbosity: bool,
):
    """
    Test that clustering by atom results in the correct information being printed
    and written to file.
    """
    element = "H"
    labels = " -1 -1\n -1 -1\n -1 -1\n"

    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = [element]

    clusterer = Clusterer(data_controller=data, verbosity=verbosity)

    clusterer.run_atom_clustering()

    assert list(clusterer._atom_environments.keys()) == [element]
    assert np.all(
        clusterer._atom_environments[element]
        == np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
                [[0.6, 1.2, 1.8], [0.7, 1.4, 2.1]],
                [[0.9, 1.8, 2.7], [1.0, 2.0, 3.0]],
            ],
            dtype="float32",
        )
    )

    file_out = "tests/data/tests_output/clustered_{}.data".format(element)
    assert isfile(file_out)
    with open(file_out) as f:
        assert f.read() == labels

    text = capsys.readouterr().out
    if verbosity >= 1:
        assert "Element: {}\n".format(element) in text
        assert "0 labels assigned\n" in text
        assert "Noise    :      " in text
        assert "Clustered in " in text
    else:
        assert text == ""


@pytest.mark.parametrize("verbosity", [True, False])
def test_run_frame_clustering(
    data: Data,
    capsys: pytest.CaptureFixture,
    verbosity: bool,
):
    """
    Test that clustering by frame results in the correct information being printed
    and written to file.
    """
    element = "H"
    labels = " -1\n -1\n -1\n"

    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = [element]

    clusterer = Clusterer(data_controller=data, verbosity=verbosity)

    clusterer.run_frame_clustering()

    assert list(clusterer._atom_environments.keys()) == [element]
    assert np.all(
        clusterer._atom_environments[element]
        == np.array(
            [
                [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
                [[0.6, 1.2, 1.8], [0.7, 1.4, 2.1]],
                [[0.9, 1.8, 2.7], [1.0, 2.0, 3.0]],
            ],
            dtype="float32",
        )
    )

    file_out = "tests/data/tests_output/clustered_frames.data"
    assert isfile(file_out)
    with open(file_out) as f:
        assert f.read() == labels

    text = capsys.readouterr().out
    if verbosity >= 1:
        assert "0 labels assigned\n" in text
        assert "Noise    :      " in text
        assert "Clustered in " in text
    else:
        assert text == ""
