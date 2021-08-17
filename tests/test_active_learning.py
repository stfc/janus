"""
Unit tests for `active_learning.py`
"""

from os import listdir, mkdir, remove, symlink
from os.path import isfile
from shutil import copy, copytree, rmtree

from genericpath import isdir
import pytest

from cc_hdnnp.active_learning import ActiveLearning
from cc_hdnnp.data import Data
from cc_hdnnp.structure import AllSpecies, AllStructures, Species, Structure


@pytest.fixture
def data() -> Data:
    """"""
    species_H = Species(symbol="H", atomic_number=1, mass=1.0)
    species_C = Species(symbol="C", atomic_number=6, mass=12.0)
    species_O = Species(symbol="O", atomic_number=8, mass=16.0)
    all_species = AllSpecies(species_H, species_C, species_O)
    structure = Structure(
        name="test", all_species=all_species, delta_E=1.0, delta_F=1.0
    )

    yield Data(
        structures=AllStructures(structure),
        main_directory="tests/data",
        scripts_sub_directory="tests_output",
        active_learning_sub_directory="tests_output",
        n2p2_bin="",
        lammps_executable="",
    )

    for file in listdir("tests/data/tests_output"):
        if isfile("tests/data/tests_output/" + file):
            remove("tests/data/tests_output/" + file)
        elif isdir("tests/data/tests_output/" + file):
            rmtree("tests/data/tests_output/" + file)


@pytest.fixture
def active_learning(data: Data) -> ActiveLearning:
    """"""
    symlink("weights.001.000020.out", "tests/data/n2p2/weights.001.data")
    symlink("weights.001.000020.out", "tests/data/n2p2/weights.006.data")
    symlink("weights.001.000020.out", "tests/data/n2p2/weights.008.data")
    copy("tests/data/n2p2/template.sh", "tests/data/tests_output/template.sh")
    copytree("tests/data/n2p2", "tests/data/n2p2_copy")

    yield ActiveLearning(
        data_controller=data,
        n2p2_directories=["tests/data/n2p2_copy", "tests/data/n2p2_copy"],
    )

    rmtree("tests/data/n2p2_copy")
    remove("tests/data/n2p2/weights.001.data")
    remove("tests/data/n2p2/weights.006.data")
    remove("tests/data/n2p2/weights.008.data")


# def prepare_active_learning(mock_active_learning: ActiveLearning):
#     """"""
#     mock_active_learning.selection = np.array([0])
#     mock_active_learning.names = np.array(["test_some_stuff"])
#     mock_active_learning.lattices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
#     mock_active_learning.elements = np.array(["H"])
#     mock_active_learning.positions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
#     mock_active_learning.charges = np.array([0])
#     mock_active_learning.statistics = np.array([[]])


def test_write_lammps_mode1_error(active_learning: ActiveLearning):
    """Test that an error is raised if the mode1 directory already exists."""
    mkdir("tests/data/tests_output/mode1")
    with pytest.raises(IOError) as e:
        active_learning.write_lammps(range(1))

    assert str(e.value) == (
        "Path mode1 already exists. Please remove old directory first if you would "
        "like to recreate it."
    )


# def test_write_lammps(active_learning: ActiveLearning):
#     """"""
#     with open("tests/data/tests_output/simulation.lammps", "w") as f:
#         f.write("run 5")
#     active_learning.write_lammps(range(1))


def test_prepare_lammps_trajectory(active_learning: ActiveLearning):
    """"""
    # Generate the mode1 directory
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")
    active_learning.write_lammps(range(1))
    symlink(
        "../../../active_learning/log.lammps",
        "tests/data/tests_output/mode1/test_npt_hdnnp1_t0_p1.0_1/log.lammps",
    )

    with pytest.raises(ValueError) as e:
        active_learning.prepare_lammps_trajectory()

    assert (
        str(e.value)
        == "tests/data/tests_output/mode1/test_npt_hdnnp1_t0_p1.0_1/log.lammps was empty"
    )


# def test_data_combine(active_learning: ActiveLearning):
#     """"""
#     prepare_active_learning(active_learning)
#     active_learning.combine_data_add()
