"""
Unit tests for `active_learning.py`
"""

from os import listdir, mkdir, remove, symlink
from os.path import isdir, isfile, join
from shutil import copy, copytree, rmtree
from typing import List

import numpy as np
import pytest

from cc_hdnnp.active_learning import ActiveLearning
from cc_hdnnp.data import Data
from cc_hdnnp.structure import AllSpecies, AllStructures, Species, Structure


# The headers used in N2P2 energy and force files
N2P2_ENERGY_HEADER = (
    "################################################################################\n"
    "# Energy comparison.\n"
    "################################################################################\n"
    "# Col  Name  Description\n"
    "################################################################################\n"
    "# 1    index Structure index.\n"
    "# 2    Eref  Reference potential energy per atom (training units).\n"
    "# 3    Ennp  NNP potential energy per atom (training units).\n"
    "################################################################################\n"
    "#        1                2                3\n"
    "#    index             Eref             Ennp\n"
    "############################################\n"
)
N2P2_FORCE_HEADER = (
    "################################################################################\n"
    "# Force comparison.\n"
    "################################################################################\n"
    "# Col  Name  Description\n"
    "################################################################################\n"
    "# 1    index_s Structure index.\n"
    "# 2    index_a Atom index (x, y, z components in consecutive lines).\n"
    "# 3    Fref    Reference force (training units).\n"
    "# 4    Fnnp    NNP force (training units).\n"
    "################################################################################\n"
    "#        1          2                3                4\n"
    "#  index_s    index_a             Fref             Fnnp\n"
    "############################################\n"
)


@pytest.fixture
def all_species() -> AllSpecies:
    """
    Fixture to create a AllSpecies object for testing.
    """
    species_H = Species(symbol="H", atomic_number=1, mass=1.0)
    species_C = Species(symbol="C", atomic_number=6, mass=12.0)
    species_O = Species(symbol="O", atomic_number=8, mass=16.0)
    return AllSpecies(species_H, species_C, species_O)


@pytest.fixture
def data(all_species: AllSpecies) -> Data:
    """
    Fixture to create a Data object for testing,
    including removal of files in the test output folder.
    """
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
    """
    Fixture to create a ActiveLearning object for testing,
    including linking and removal of relevant files.
    """
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


def prepare_mode1_files(files: List[str] = None):
    """ """
    mkdir("tests/data/tests_output/mode1")
    mkdir("tests/data/tests_output/mode1/test_npt_hdnnp1_t325_p1.0_1")
    for f in listdir("tests/data/active_learning/mode1/test_npt_hdnnp1_t325_p1.0_1"):
        if files is not None and f not in files:
            continue
        copy(
            join("tests/data/active_learning/mode1/test_npt_hdnnp1_t325_p1.0_1", f),
            "tests/data/tests_output/mode1/test_npt_hdnnp1_t325_p1.0_1",
        )


def prepare_mode2_files():
    """ """
    mkdir("tests/data/tests_output/mode2")
    mkdir("tests/data/tests_output/mode2/HDNNP_1")
    mkdir("tests/data/tests_output/mode2/HDNNP_2")

    with open("tests/data/tests_output/mode2/HDNNP_1/mode_2.out", "w") as f:
        f.write("TIMING Training loop finished: 3661 s")
    with open("tests/data/tests_output/mode2/HDNNP_2/mode_2.out", "w") as f:
        f.write("TIMING Training loop finished: 61 s")

    with open("tests/data/tests_output/mode2/HDNNP_1/trainpoints.000000.out", "w") as f:
        f.write(N2P2_ENERGY_HEADER + "0 0 10")
    with open("tests/data/tests_output/mode2/HDNNP_2/trainpoints.000000.out", "w") as f:
        f.write(N2P2_ENERGY_HEADER + "0 0 20")

    with open("tests/data/tests_output/mode2/HDNNP_1/trainforces.000000.out", "w") as f:
        f.write(N2P2_FORCE_HEADER + "0 0 0 10\n0 0 0 10\n0 0 0 10")
    with open("tests/data/tests_output/mode2/HDNNP_2/trainforces.000000.out", "w") as f:
        f.write(N2P2_FORCE_HEADER + "0 0 0 20\n0 0 0 20\n0 0 0 20")


# TEST WRITE LAMMPS


def test_write_lammps_mode1_error(active_learning: ActiveLearning):
    """Test that an error is raised if the mode1 directory already exists."""
    mkdir("tests/data/tests_output/mode1")
    with pytest.raises(IOError) as e:
        active_learning.write_lammps(range(1))

    assert str(e.value) == (
        "Path mode1 already exists. Please remove old directory first if you would "
        "like to recreate it."
    )


def test_write_lammps(active_learning: ActiveLearning):
    """
    Test the `write_lammps` function is successful,
    leading to the creation of empty joblist_mode1.dat.
    """
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")
    active_learning.write_lammps(range(1))
    assert isfile("tests/data/tests_output/joblist_mode1.dat")
    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp1_t0_p1.0_1/input.lammps"
    )
    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp2_t0_p1.0_2/input.lammps"
    )


def test_write_lammps_nonzero_joblist(active_learning: ActiveLearning):
    """
    Test the `write_lammps` function is successful when `max_joblist_len` is non-zero,
    not leading to the creation of empty joblist_mode1.dat.
    """
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")
    active_learning.max_len_joblist = 1
    active_learning.write_lammps(range(1))
    assert not isfile("tests/data/tests_output/joblist_mode1.dat")
    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp1_t0_p1.0_1/input.lammps"
    )
    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp2_t0_p1.0_2/input.lammps"
    )


def test_write_lammps_name_is_none(
    all_species: AllSpecies, active_learning: ActiveLearning
):
    """
    Test the `write_lammps` function is successful when the name of one of the structures is
    `None`.
    """
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")

    structure = Structure(name=None, all_species=all_species, delta_E=1.0, delta_F=1.0)
    active_learning.all_structures = AllStructures(structure)

    active_learning.write_lammps(range(1))
    assert isfile("tests/data/tests_output/joblist_mode1.dat")
    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp1_t0_p1.0_1/input.lammps"
    )
    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp2_t0_p1.0_2/input.lammps"
    )


def test_write_lammps_nve(active_learning: ActiveLearning):
    """
    Test the `write_lammps` function is successful when using the "nve" integrator.
    """
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")

    active_learning.integrators = ["nve"]

    active_learning.write_lammps(range(1))
    assert isfile("tests/data/tests_output/joblist_mode1.dat")
    assert isfile("tests/data/tests_output/mode1/test_nve_hdnnp1_t0_1/input.lammps")
    assert isfile("tests/data/tests_output/mode1/test_nve_hdnnp2_t0_2/input.lammps")


def test_write_lammps_selection(
    all_species: AllSpecies,
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
):
    """
    Test the `write_lammps` function prints a warning when using selection rate greater than 1.
    Also test that an `IndexError` if the selection exceeds the number of structures provided.
    """
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")

    structure = Structure(
        name="test", all_species=all_species, delta_E=1.0, delta_F=1.0, selection=[0, 2]
    )
    active_learning.all_structures = AllStructures(structure)

    with pytest.raises(IndexError) as e:
        active_learning.write_lammps(range(1))

    assert isfile("tests/data/tests_output/joblist_mode1.dat")
    assert not isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp1_t0_p1.0_1/input.lammps"
    )
    assert not isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp2_t0_p1.0_2/input.lammps"
    )
    assert (
        capsys.readouterr().out
        == "Starting from the 0th structure every 2th structure of "
        + "the input.data file is used.\n"
        + "The given variations of the settings are repeated 1 times\n"
        + "WARNING: The structures of the input.data file are used more than once.\n"
        + "Try to avoid this by start from the 1th structure "
        + "and using again every 2th structure.\n"
    )
    assert (
        str(e.value)
        == "`names`=['test'] does not not have entries for the `selection`=1"
    )


def test_write_lammps_comment_name_keyword_none(
    all_species: AllSpecies, active_learning: ActiveLearning
):
    """
    Test the `write_lammps` function is successful, and file paths do not include names if
    `comment_name_keyword` is `None`.
    """
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")

    structure = Structure(name=None, all_species=all_species, delta_E=1.0, delta_F=1.0)
    active_learning.all_structures = AllStructures(structure)
    active_learning.comment_name_keyword = None
    active_learning.write_lammps(range(1))

    assert isfile("tests/data/tests_output/joblist_mode1.dat")
    assert isfile("tests/data/tests_output/mode1/npt_hdnnp1_t0_p1.0_1/input.lammps")
    assert isfile("tests/data/tests_output/mode1/npt_hdnnp2_t0_p1.0_2/input.lammps")


def test_prepare_lammps_trajectory(active_learning: ActiveLearning):
    """ """
    # Generate the mode1 directory
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")
    prepare_mode1_files(files=["log.lammps", "structure.lammpstrj"])

    active_learning.prepare_lammps_trajectory()

    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp1_t325_p1.0_1/extrapolation.dat"
    )


def test_prepare_data_new(
    active_learning: ActiveLearning, capsys: pytest.CaptureFixture
):
    """ """
    prepare_mode1_files()

    active_learning.prepare_data_new()

    assert capsys.readouterr().out == (
        "Structure: test\n"
        "WARNING: Extrapolation occurred already in the first time step in "
        "test_npt_hdnnp1_t325_p1.0_1.\n"
        "Only less than 0.1% of the simulations show an extrapolation if a tolerance of 0.01 "
        "is employed (the initial 20 time steps are neglected). The tolerance value is "
        "reduced to 0.001.\n"
        "Small extrapolations are present in 1 of 1 simulations (100.0%).\n"
        "There are no large extrapolations for a tolerance of 0.1 (the initial 20 time steps "
        "are neglected). The tolerance value is reduced to 0.075.\n"
        "There are no large extrapolations for a tolerance of 0.075 (the initial 20 time "
        "steps are neglected). The tolerance value is reduced to 0.05.\n"
        "There are no large extrapolations for a tolerance of 0.05 (the initial 20 time steps "
        "are neglected). The tolerance value is reduced to 0.025.\n"
        "The median number of time steps to a small extrapolation is 1 (HDNNP_1: 1, HDNNP_2: "
        "nan).\n"
        "The median number of time steps between the first and second selected extrapolated "
        "structure is 6 (HDNNP_1: 6, HDNNP_2: nan).\n"
        "Writing 1 names to input.data-new\n"
        "Batch script written to tests/data/tests_output/active_learning_nn.sh\n"
    )


def test_prepare_data_add(
    active_learning: ActiveLearning, capsys: pytest.CaptureFixture
):
    """"""
    prepare_mode2_files()

    active_learning.lattices = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    active_learning.elements = np.array([["H"]])
    active_learning.charges = np.array([[0]])
    active_learning.names = np.array(["test_s0"])
    active_learning.positions = np.array([[[0, 0, 0]]])
    active_learning.statistics = np.array([["small", "H", "0", "0"]])
    active_learning.prepare_data_add()

    assert (
        capsys.readouterr().out
        == "\nTime to calculate 1 structures using RuNNer: "
        + "HDNNP_1: 1.02 h, HDNNP_2: 1.02 min.\n\n"
        + "1 structures identified over energy threshold `dE=[1.]`\n"
        + "1 structures identified over force threshold `dF=[1. 1. 1.]`\n"
        + "1 missing structures were identified.\n"
        + "1 missing structures originate from small extrapolations.\n"
        + "0 missing structures originate from large extrapolations.\n"
    )


def test_prepare_data_add_error(active_learning: ActiveLearning):
    """
    Test that an error is raised when calling `prepare_data_add` without a `mode2` directory.
    """

    with pytest.raises(OSError) as e:
        active_learning.prepare_data_add()

    assert str(e.value) == "`mode2` directory not found."


def test_print_statistics_no_selection(
    active_learning: ActiveLearning, capsys: pytest.CaptureFixture
):
    """ """
    prepare_mode2_files()
    statistics = np.array([[]])
    selection = np.array([0])
    names = np.array(["test_s0"])

    active_learning._print_statistics(
        selection=selection, statistics=statistics, names=names
    )

    assert capsys.readouterr().out + "1 missing structures were identified.\n"


def test_print_statistics_multiple_names(
    all_species: AllSpecies,
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
):
    """ """
    prepare_mode2_files()

    structure_0 = Structure(
        name="test0", all_species=all_species, delta_E=1.0, delta_F=1.0
    )
    structure_1 = Structure(
        name="test1", all_species=all_species, delta_E=1.0, delta_F=1.0
    )
    active_learning.all_structures = AllStructures(structure_0, structure_1)
    statistics = np.array([["small", "H", "0", "0"]])
    selection = np.array([0])
    names = np.array(["test0_s0", "test1_s0"])

    active_learning._print_statistics(
        selection=selection, statistics=statistics, names=names
    )

    assert (
        capsys.readouterr().out
        == "Structure: test0\n"
        + "1 missing structures were identified.\n"
        + "1 missing structures originate from small extrapolations.\n"
        + "0 missing structures originate from large extrapolations.\n"
        + "Structure: test1\n"
        + "0 missing structures were identified.\n"
    )


def test_analyse_extrapolation_statistics_multiple_elements(
    active_learning: ActiveLearning,
):
    """
    Test that multiple files are created when multiple elements present in the `statistics`.
    """
    statistics = np.array([["small", "H, C", "0, 1", "0, 1"]])

    active_learning._analyse_extrapolation_statistics(statistics=statistics)

    assert isfile("tests/data/tests_output/extrapolation_statistics_H.dat")
    assert isfile("tests/data/tests_output/extrapolation_statistics_C.dat")


def test_improve_selection_all_extrapolated_structures_false(
    all_species: AllSpecies, active_learning: ActiveLearning
):
    """
    Test that `_improve_selection` returns a selection when `all_extrapolated_structures` is
    `False`.
    """
    structure = Structure(
        name="test",
        all_species=all_species,
        delta_E=1.0,
        delta_F=1.0,
        all_extrapolated_structures=False,
    )
    all_structures = AllStructures(structure)
    statistics = np.array([["small", "H", "0", "0"]])
    selection = np.array([0])
    names = np.array(["test_s0"])
    ordered_structure_names = np.array(["test"])
    active_learning.all_structures = all_structures

    selection = active_learning._improve_selection(
        selection=selection,
        statistics=statistics,
        names=names,
        ordered_structure_names=ordered_structure_names,
    )

    assert selection == np.array([0])


def test_improve_selection_all_empty_statistics(
    all_species: AllSpecies, active_learning: ActiveLearning
):
    """
    Test that `_improve_selection` returns a selection when `statistics` is empty.
    """
    structure = Structure(
        name="test",
        all_species=all_species,
        delta_E=1.0,
        delta_F=1.0,
        all_extrapolated_structures=False,
    )
    all_structures = AllStructures(structure)
    statistics = np.array([[], []])
    selection = np.array([0, 1])
    names = np.array(["test_s0", "test_s1"])
    ordered_structure_names = np.array(["test", "test"])
    active_learning.all_structures = all_structures

    selection = active_learning._improve_selection(
        selection=selection,
        statistics=statistics,
        names=names,
        ordered_structure_names=ordered_structure_names,
    )

    assert all(selection == np.array([0, 1]))


def test_improve_selection_reduce_selection(
    all_species: AllSpecies, active_learning: ActiveLearning
):
    """
    Test that `_improve_selection` reduces the selection (calls `_reduce_selection`) when steps
    is larger than 2.

    For both "test0" and "test1", there should only be 2 of the corresponding steps in the
    `selection`, reduced from the initial three each.
    """
    structure_0 = Structure(
        name="test0",
        all_species=all_species,
        delta_E=1.0,
        delta_F=1.0,
        all_extrapolated_structures=False,
    )
    structure_1 = Structure(
        name="test1",
        all_species=all_species,
        delta_E=1.0,
        delta_F=1.0,
        all_extrapolated_structures=False,
    )
    all_structures = AllStructures(structure_0, structure_1)
    statistics = np.array([[], [], [], [], [], [], []])
    selection = np.array([0, 1, 2, 3, 4, 5])
    names = np.array(
        [
            "test0_s0",
            "test0_s1",
            "test0_s2",
            "test1_s3",
            "test1_s4",
            "test1_s5",
            "test1_s6",
        ]
    )
    ordered_structure_names = np.array(
        ["test0", "test0", "test0", "test1", "test1", "test1", "test1"]
    )
    active_learning.all_structures = all_structures
    structures = list(all_structures.structure_dict.values())
    active_learning._validate_timesteps(
        active_learning.timestep, active_learning.N_steps, structures
    )

    selection = active_learning._improve_selection(
        selection=selection,
        statistics=statistics,
        names=names,
        ordered_structure_names=ordered_structure_names,
    )

    assert all(selection == np.array([0, 2, 3, 5]))


def test_improve_selection_max_extrapolated_structures(
    all_species: AllSpecies,
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
):
    """
    Test that `_improve_selection` handles cases where we exceed max_extrapolated_structures.
    """
    structure = Structure(
        name="test",
        all_species=all_species,
        delta_E=1.0,
        delta_F=1.0,
        max_extrapolated_structures=1,
    )
    all_structures = AllStructures(structure)
    statistics = np.array([["small", "H", "0", "0"], ["small", "H", "0", "0"]])
    selection = np.array([0])
    names = np.array(["test_s0", "test_s0"])
    ordered_structure_names = np.array(["test", "test"])
    active_learning.all_structures = all_structures
    structures = list(all_structures.structure_dict.values())
    active_learning._validate_timesteps(
        active_learning.timestep, active_learning.N_steps, structures
    )

    selection = active_learning._improve_selection(
        selection=selection,
        statistics=statistics,
        names=names,
        ordered_structure_names=ordered_structure_names,
    )

    assert selection == np.array([0]) or selection == np.array([1])
    assert capsys.readouterr().out == "The extrapolation ['H', '0'] occurred 2 times.\n"


def test_improve_selection_exceptions(
    all_species: AllSpecies, active_learning: ActiveLearning
):
    """
    Test that `_improve_selection` handles cases where `exceptions` is set.
    If the final entry is 0, then all matching extrapolations should be removed.
    """
    structure = Structure(
        name="test",
        all_species=all_species,
        delta_E=1.0,
        delta_F=1.0,
        exceptions=[["H", "0", 0]],
    )
    all_structures = AllStructures(structure)
    statistics = np.array([["small", "H", "0", "0"]])
    selection = np.array([0])
    names = np.array(["test_s0"])
    ordered_structure_names = np.array(["test"])
    active_learning.all_structures = all_structures
    structures = list(all_structures.structure_dict.values())
    active_learning._validate_timesteps(
        active_learning.timestep, active_learning.N_steps, structures
    )

    selection = active_learning._improve_selection(
        selection=selection,
        statistics=statistics,
        names=names,
        ordered_structure_names=ordered_structure_names,
    )

    assert len(selection) == 0


def test_reduce_selection_min_separation(active_learning: ActiveLearning):
    """
    Test that `_reduce_selection` reduces the selection in the case where
    `t_separation_interpolation_checks` is not greater than the smallest step seperation.
    """
    selection = np.array([0])

    selection = active_learning._reduce_selection(
        selection=selection,
        max_interpolated_structures_per_simulation=4,
        t_separation_interpolation_checks=1,
        steps=[0, 1, 2],
        indices=[0, 0, 0],
    )

    assert len(selection) == 0


def test_read_forces_point_format(active_learning: ActiveLearning):
    """
    Test that `_read_forces` raises and error when given a file in point format.
    """
    prepare_mode2_files()
    with open("tests/data/tests_output/mode2/HDNNP_1/trainforces.000000.out", "w") as f:
        f.write("point\n1 0 0 0 10\n1 0 0 0 10\n1 0 0 0 10")

    forces = active_learning._read_forces(
        input_name="tests/data/tests_output/mode2/HDNNP_1/trainforces.000000.out"
    )

    assert len(forces) == 3
    assert all(forces[0] == np.array([0.0, 10.0]))
    assert all(forces[1] == np.array([0.0, 10.0]))
    assert all(forces[2] == np.array([0.0, 10.0]))


def test_read_forces_conf_format(active_learning: ActiveLearning):
    """
    Test that `_read_forces` raises and error when given a file in Conf. format.
    """
    prepare_mode2_files()
    with open("tests/data/tests_output/mode2/HDNNP_1/trainforces.000000.out", "w") as f:
        f.write("Conf.\n1 0 0 0 0 10 10 10")

    forces = active_learning._read_forces(
        input_name="tests/data/tests_output/mode2/HDNNP_1/trainforces.000000.out"
    )

    assert len(forces) == 3
    assert all(forces[0] == np.array([0.0, 10.0]))
    assert all(forces[1] == np.array([0.0, 10.0]))
    assert all(forces[2] == np.array([0.0, 10.0]))


def test_read_forces_unknown_format(active_learning: ActiveLearning):
    """
    Test that `_read_forces` raises and error when given a file in an unknown format.
    """
    prepare_mode2_files()
    with open("tests/data/tests_output/mode2/HDNNP_1/trainforces.000000.out", "w") as f:
        f.write("Unrecognised format")

    with pytest.raises(OSError) as e:
        active_learning._read_forces(
            input_name="tests/data/tests_output/mode2/HDNNP_1/trainforces.000000.out"
        )

    assert str(e.value) == "Unknown RuNNer format"


def test_read_energy_point_format(active_learning: ActiveLearning):
    """
    Test that `_read_energy` raises and error when given a file in point format.
    """
    prepare_mode2_files()
    with open("tests/data/tests_output/mode2/HDNNP_1/trainpoints.000000.out", "w") as f:
        f.write("point\n1 0 10")

    energy = active_learning._read_energies(
        input_name="tests/data/tests_output/mode2/HDNNP_1/trainpoints.000000.out"
    )

    assert all(energy[0] == np.array([0.0, 10.0]))


def test_read_energy_conf_format(active_learning: ActiveLearning):
    """
    Test that `_read_energy` raises and error when given a file in Conf. format.
    """
    prepare_mode2_files()
    with open("tests/data/tests_output/mode2/HDNNP_1/trainpoints.000000.out", "w") as f:
        f.write("Conf.\n1 1 0 10")

    energy = active_learning._read_energies(
        input_name="tests/data/tests_output/mode2/HDNNP_1/trainpoints.000000.out"
    )

    assert all(energy[0] == np.array([0.0, 10.0]))


def test_read_energy_unknown_format(active_learning: ActiveLearning):
    """
    Test that `_read_energy` raises and error when given a file in an unknown format.
    """
    prepare_mode2_files()
    with open("tests/data/tests_output/mode2/HDNNP_1/trainpoints.000000.out", "w") as f:
        f.write("Unrecognised format")

    with pytest.raises(OSError) as e:
        active_learning._read_energies(
            input_name="tests/data/tests_output/mode2/HDNNP_1/trainpoints.000000.out"
        )

    assert str(e.value) == "Unknown RuNNer format"


def test_read_data(active_learning: ActiveLearning):
    """
    Test that `_read_data` extracts all information correctly.
    """

    active_learning._read_data("tests/data/n2p2/input.data")

    assert active_learning.names == np.array(["test_s0"])
    np.testing.assert_allclose(
        active_learning.lattices,
        np.array(
            [
                [
                    [17.71284412, 0.0, 0.0],
                    [0.0, 17.71284412, 0.0],
                    [0.0, 0.0, 17.71284412],
                ]
            ]
        ),
    )
    assert active_learning.elements.shape == (1, 512)
    assert active_learning.positions.shape == (1, 512, 3)
    assert active_learning.charges.shape == (1, 512)
    assert active_learning.statistics == np.array(["stats"])


def test_print_reliability(active_learning: ActiveLearning):
    """
    Test that `_print_reliability` extracts all information correctly.
    """

    active_learning._read_data("tests/data/n2p2/input.data")

    assert active_learning.names == np.array(["test_s0"])
    np.testing.assert_allclose(
        active_learning.lattices,
        np.array(
            [
                [
                    [17.71284412, 0.0, 0.0],
                    [0.0, 17.71284412, 0.0],
                    [0.0, 0.0, 17.71284412],
                ]
            ]
        ),
    )
    assert active_learning.elements.shape == (1, 512)
    assert active_learning.positions.shape == (1, 512, 3)
    assert active_learning.charges.shape == (1, 512)
    assert active_learning.statistics == np.array(["stats"])
