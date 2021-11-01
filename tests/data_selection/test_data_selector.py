"""
Unit tests for `dataselector.py`
"""

from os import listdir, remove
from os.path import isfile
from shutil import rmtree
from typing import List

from genericpath import isdir
import pytest

from cc_hdnnp.data import Data
from cc_hdnnp.data_selection.dataselector import DataSelector
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


@pytest.mark.parametrize(
    "n_atoms, error",
    [
        (
            [0, 1],
            "Datasets containing a varying number of atoms per frame are not supported, "
            + "datset contained [0 1] atoms in different frames.",
        ),
        (
            [],
            "Dataset contains no frames.",
        ),
    ],
)
def test_run_frame_clustering_errors(
    data: Data,
    n_atoms: List[int],
    error: str,
):
    """
    Test that errors are raised if the number of atoms per frame is incorrect.
    """
    data.n2p2_directories = ["tests/data/tests_output"]
    with open("tests/data/tests_output/input.data", "w") as f:
        for n in n_atoms:
            f.write("begin\nlattice 1 0 0\nlattice 0 1 0\nlattice 0 0 1\n")
            [f.write("atom    0.0 0.0 0.0 H 0.0 0.0 0.0 0.0 0.0\n") for _ in range(n)]
            f.write("energy 0\ncharge 0\nend\n")

    with pytest.raises(ValueError) as e:
        DataSelector(data_controller=data)

    assert str(e.value) == error
