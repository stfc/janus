"""
Unit tests for `active_learning.py`
"""

from os import listdir, mkdir, remove, symlink
from os.path import isdir, isfile, join
from shutil import copy, copytree, rmtree
from typing import List, Tuple, Union

import numpy as np
import pytest

from janus.active_learning import ActiveLearning
from janus.controller import Controller
from janus.dataset import Dataset, Frame
from janus.structure import AllStructures, Species, Structure


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
PREPARE_DATA_NEW_STDOUT = (
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
)


@pytest.fixture
def all_species() -> List[Species]:
    """
    Fixture to create a Structure object for testing.
    """
    species_H = Species(symbol="H", atomic_number=1, mass=1.0)
    species_C = Species(symbol="C", atomic_number=6, mass=12.0)
    species_O = Species(symbol="O", atomic_number=8, mass=16.0)
    return [species_H, species_C, species_O]


@pytest.fixture
def controller(all_species: List[Species]) -> Controller:
    """
    Fixture to create a Data object for testing,
    including removal of files in the test output folder.
    """
    structure = Structure(
        name="test", all_species=all_species, delta_E=1.0, delta_F=1.0
    )

    yield Controller(
        structures=AllStructures(structure),
        main_directory="tests/data",
        scripts_sub_directory="tests_output",
        active_learning_sub_directory="tests_output",
        n2p2_bin="",
        lammps_executable="",
        n2p2_sub_directories=["n2p2_copy", "n2p2_copy"],
    )

    for file in listdir("tests/data/tests_output"):
        if isfile("tests/data/tests_output/" + file):
            remove("tests/data/tests_output/" + file)
        elif isdir("tests/data/tests_output/" + file):
            rmtree("tests/data/tests_output/" + file)


@pytest.fixture
def active_learning(controller: Controller) -> ActiveLearning:
    """
    Fixture to create a ActiveLearning object for testing,
    including linking and removal of relevant files.
    """
    if not isfile("tests/data/n2p2/weights.001.data"):
        symlink("weights.001.000020.out", "tests/data/n2p2/weights.001.data")
    if not isfile("tests/data/n2p2/weights.006.data"):
        symlink("weights.001.000020.out", "tests/data/n2p2/weights.006.data")
    if not isfile("tests/data/n2p2/weights.008.data"):
        symlink("weights.001.000020.out", "tests/data/n2p2/weights.008.data")
    copy("tests/data/scripts/template.sh", "tests/data/tests_output/template.sh")
    copytree("tests/data/n2p2", "tests/data/n2p2_copy")

    yield ActiveLearning(
        data_controller=controller,
    )

    rmtree("tests/data/n2p2_copy")
    remove("tests/data/n2p2/weights.001.data")
    remove("tests/data/n2p2/weights.006.data")
    remove("tests/data/n2p2/weights.008.data")


@pytest.fixture
def basic_dataset(active_learning: ActiveLearning) -> Dataset:
    """
    Fixture to create a basic Dataset object from a single Frame containing
    only H atoms.
    """
    frame = Frame(
        name="test",
        lattice=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        symbols=np.array(["H" for _ in range(512)]),
        charges=np.array([0.0 for _ in range(512)]),
        positions=np.array([[0.0, 0.0, 0.0] for _ in range(512)]),
        statistics=np.array(["small", "H", "0", "0"]),
    )
    return Dataset(frames=[frame], all_structures=active_learning.all_structures)


def prepare_mode1_files(files: List[str] = None):
    """Mocks the mode1 files for testing."""
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
    """Mocks the mode2 files for testing."""
    copytree("tests/data/active_learning/mode2", "tests/data/tests_output/mode2")

    with open("tests/data/tests_output/mode2/HDNNP_1/mode_2.out", "w") as f:
        f.write("TIMING Training loop finished: 3661 s")
    with open("tests/data/tests_output/mode2/HDNNP_2/mode_2.out", "w") as f:
        f.write("TIMING Training loop finished: 61 s")


# TEST INIT


def test_init_len_n2p2_directories(controller: Controller):
    """Test that a ValueError is raised for an empty list of `n2p2_directories`."""
    controller.n2p2_directories = []
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
        )

    assert (
        str(e.value)
        == "`data_controller.n2p2_directories` must have 2 entries, but had 0"
    )


def test_init_integrators(controller: Controller):
    """Test that unrecognised `integrators` raise a value error."""
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            integrators=["unrecognised"],
        )

    assert str(e.value) == "Integrator unrecognised is not implemented."


def test_init_npt_no_pressures(controller: Controller):
    """Test that not providing pressures raises an error for the "npt" integrator."""
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            pressures=[],
        )

    assert (
        str(e.value)
        == "Integrator npt requires to specify at least one value for pressure."
    )


def test_init_barostat_option(controller: Controller):
    """Test that unrecognised `barostat_option` raise a value error."""
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            barostat_option="unrecognised",
        )

    assert str(e.value) == "Barostat option unrecognised is not implemented."


def test_init_atom_style(controller: Controller):
    """Test that unrecognised `atom_style` raise a value error."""
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            atom_style="unrecognised",
        )

    assert str(e.value) == "Atom style unrecognised is not implemented."


def test_init_N_steps(controller: Controller):
    """
    Test that a ValueError is raised when `N_steps` is not a multiple of `dump_lammpstrj`.
    """
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            N_steps=200001,
        )

    assert (
        str(e.value)
        == "N_steps has to be a multiple of dump_lammpstrj (200001!=N*200)."
    )


def test_init_dump_lammpstrj(controller: Controller):
    """
    Test that when `dump_lammpstrj` is less than the `min_t_separation_interpolation` of a
    Structure then a ValueError is raised.
    """
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            dump_lammpstrj=1,
        )

    assert str(e.value) == (
        "The extrapolation free structures would be stored only every 1th time step, "
        "but the minimum time step separation of interpolated structures are set to "
        "[200] time steps."
    )


def test_init_max_len_joblist(controller: Controller):
    """Test that a negative values for `max_len_joblist` raises a ValueError."""
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            max_len_joblist=-1,
        )

    assert str(e.value) == (
        "The maximal length of the job list has to be set to 0 (which means infinity) "
        "or a positive integer number, but it was -1"
    )


def test_init_comment_name_keyword(controller: Controller):
    """
    Test that setting `comment_name_keyword` to `None` raises a ValueError when
    the `name` of a Structure is not `None`.
    """
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            comment_name_keyword=None,
        )

    assert str(e.value) == (
        "If comment_name_keyword or structure_names is set to None the other one has to"
        " be set to None as well."
    )


def test_init_structure_names_none(all_species: List[Species]):
    """
    Test that using a combination of str and `None` for Structure `name`s raises a TypeError.
    """
    structure_0 = Structure(
        name="test", all_species=all_species, delta_E=1.0, delta_F=1.0
    )
    structure_1 = Structure(
        name=None, all_species=all_species, delta_E=1.0, delta_F=1.0
    )
    controller = Controller(
        structures=AllStructures(structure_0, structure_1),
        main_directory="tests/data",
        scripts_sub_directory="tests_output",
        active_learning_sub_directory="tests_output",
        n2p2_bin="",
        lammps_executable="",
        n2p2_sub_directories=["n2p2_copy", "n2p2_copy"],
    )

    with pytest.raises(TypeError) as e:
        ActiveLearning(
            data_controller=controller,
        )

    assert str(e.value) == (
        "Individual structure names cannot be set to None when using multiple Structures."
    )


def test_init_initial_tolerance(controller: Controller):
    """Test that values of `initial_tolerance` <= 1 rause a ValueError."""
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            initial_tolerance=1,
        )

    assert str(e.value) == ("The value of initial_tolerance has to be higher than 1.")


def test_init_len_tolerances(controller: Controller):
    """Test that a value of `initial_tolerance` greater than the length of `tolerances` raises
    a ValueError."""
    with pytest.raises(ValueError) as e:
        ActiveLearning(
            data_controller=controller,
            initial_tolerance=100,
        )

    assert str(e.value) == (
        "There are not enough tolerance values as initial_tolerance results in an index "
        "error."
    )


@pytest.mark.parametrize("mode1", [True, False])
@pytest.mark.parametrize("mode2", [True, False])
@pytest.mark.parametrize("new", [True, False])
@pytest.mark.parametrize("add", [True, False])
def test_init_remove_existing_files(
    controller: Controller,
    mode1: bool,
    mode2: bool,
    new: bool,
    add: bool,
):
    """Test that old files are removed if `remove_existing_files` is True."""
    if mode1:
        mkdir("tests/data/tests_output/mode1")
    if mode2:
        mkdir("tests/data/tests_output/mode2")
    if new:
        open("tests/data/tests_output/input.data-new", "w")
    if add:
        open("tests/data/tests_output/input.data-add", "w")

    ActiveLearning(
        data_controller=controller,
        remove_existing_files=True,
    )

    assert not isdir("tests/data/tests_output/mode1")
    assert not isdir("tests/data/tests_output/mode2")
    assert not isfile("tests/data/tests_output/input.data-new")
    assert not isfile("tests/data/tests_output/input.data-add")


# TEST VALIDATE TIMESTEPS


def test_write_validate_timesteps_warning(
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
):
    """Test that a warning is printed for timesteps larger than 0.01 ps."""
    active_learning._validate_timesteps(
        timestep=0.02,
        N_steps=active_learning.N_steps,
        structures=active_learning.all_structures.values(),
    )

    assert capsys.readouterr().out == "WARNING: Very large timestep of 0.02 ps.\n"


def test_write_validate_timesteps_N_steps(active_learning: ActiveLearning):
    """Test that a ValueError is raised if `t_separation_interpolation_checks` is larger than
    a fifth of the totoal number of steps `N_steps`."""
    with pytest.raises(ValueError) as e:
        active_learning._validate_timesteps(
            timestep=0.02,
            N_steps=1,
            structures=active_learning.all_structures.values(),
        )

    assert str(e.value) == (
        "`t_separation_interpolation_checks=10000` must be less than a fifth of "
        "`N_steps=1` for all structures"
    )


def test_write_validate_timesteps_min_t_separation_interpolation(
    all_species: List[Species], active_learning: ActiveLearning
):
    """Test that if `t_separation_interpolation_checks` is less than
    `min_t_separation_interpolation`, a ValueError is raised."""
    structure = Structure(
        name=None,
        all_species=all_species,
        delta_E=1.0,
        delta_F=1.0,
        t_separation_interpolation_checks=0,
    )
    with pytest.raises(ValueError) as e:
        active_learning._validate_timesteps(
            timestep=active_learning.timestep,
            N_steps=active_learning.N_steps,
            structures=[structure],
        )

    assert str(e.value) == (
        "`t_separation_interpolation_checks` must be equal to or greater than "
        "`min_t_separation_interpolation` for all structures"
    )


# TEST READ INPUT DATA


@pytest.mark.parametrize(
    "periodic, directories",
    [
        (
            False,
            ["tests/data/n2p2/no_lattice", "tests/data/n2p2/no_lattice"],
        )
    ],
)
def test_read_input_data(
    active_learning: ActiveLearning, periodic: bool, directories: List[str]
):
    """
    Test that the Dataset returned by `_read_input_data` has the expected
    values, notably that the lattice is correctly created for a non-periodic structure.
    """
    active_learning.periodic = periodic
    active_learning.data_controller.n2p2_directories = directories
    with open("tests/data/tests_output/input.data", "w") as f:
        f.write("different file")

    dataset = active_learning._read_input_data()

    assert dataset.all_names == np.array(["test"])
    assert dataset.all_lattices.shape == (1, 3, 3)
    np.testing.assert_allclose(
        dataset.all_lattices,
        np.array([[[30.280816, 0, 0], [0, 30.304311, 0], [0, 0, 30.289951]]]),
    )
    assert dataset.all_symbols.shape == (1, 512)
    assert dataset.all_positions.shape == (1, 512, 3)
    assert dataset.all_charges.shape == (1, 512)


@pytest.mark.parametrize(
    "periodic, directories, error",
    [
        (
            True,
            ["tests/data/n2p2_copy", "tests/data/n2p2/no_lattice"],
            (
                "input.data files in tests/data/n2p2_copy and tests/data/n2p2/no_lattice "
                "are differnt."
            ),
        ),
        (
            True,
            ["tests/data/n2p2/no_atoms", "tests/data/n2p2/no_atoms"],
            "For some of the structures the definition of the atoms is incomplete or missing.",
        ),
        (
            True,
            ["tests/data/n2p2/no_lattice", "tests/data/n2p2/no_lattice"],
            (
                "The periodic keyword is set to True but for some of the structures the "
                "definition of the lattice is incomplete or missing."
            ),
        ),
        (
            False,
            ["tests/data/n2p2_copy", "tests/data/n2p2_copy"],
            (
                "The periodic keyword is set to False but for some of the "
                "structures a definition of a lattice exists."
            ),
        ),
    ],
)
def test_read_input_data_errors(
    active_learning: ActiveLearning, periodic: bool, directories: List[str], error: str
):
    """Test that errors are raised if bad data is provided."""
    active_learning.periodic = periodic
    active_learning.data_controller.n2p2_directories = directories
    with open("tests/data/tests_output/input.data", "w") as f:
        f.write("different file")
    with pytest.raises(ValueError) as e:
        active_learning._read_input_data()

    assert str(e.value) == error


# TEST WRITE INPUT LAMMPS


@pytest.mark.parametrize("is_orthorhombic", [True, False])
@pytest.mark.parametrize(
    "periodic, periodic_dump", [(True, " x y z"), (False, " xu yu zu")]
)
@pytest.mark.parametrize("atom_style, charge_dump", [("atomic", ""), ("full", " q")])
def test_write_input_lammps(
    active_learning: ActiveLearning,
    periodic: bool,
    periodic_dump: str,
    atom_style: str,
    charge_dump: str,
    is_orthorhombic: bool,
):
    """
    Test the `_input_write_lammps` function is successful for both `atom_style` values and
    lattice periodicites, leading to differences in the dump command.
    """
    output_directory = "tests/data/tests_output"
    active_learning.periodic = periodic
    active_learning.atom_style = atom_style
    with open(output_directory + "/simulation.lammps", "w") as f:
        f.write("run 5")

    active_learning._write_input_lammps(
        path=output_directory,
        seed=1,
        temperature=1,
        pressure=1.0,
        integrator="npt",
        is_orthorhombic=is_orthorhombic,
    )

    assert isfile(output_directory + "/input.lammps")
    with open(output_directory + "/input.lammps") as f:
        text = f.read()
    assert (
        "dump lammpstrj all custom 1 structure.lammpstrj id element{0}{1}"
        "".format(periodic_dump, charge_dump)
    ) in text
    if is_orthorhombic:
        assert "fix z all npt temp 1 1 0.05 iso" in text
    else:
        assert "fix z all npt temp 1 1 0.05 tri" in text


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize(
    "atom_style, integrator, error",
    [
        (
            "atomic",
            "unrecognised",
            "`integrator` must be one of 'nve', 'nvt' or 'npt', but was 'unrecognised'",
        ),
        ("unrecognised", "npt", "Atom style unrecognised is not implemented."),
    ],
)
def test_write_input_lammps_errors(
    active_learning: ActiveLearning,
    periodic: bool,
    atom_style: str,
    integrator: str,
    error: str,
):
    """
    Test the `_input_write_lammps` function is successful for both `atom_style` values and
    lattice periodicities, leading to differences in the dump command.
    """
    output_directory = "tests/data/tests_output"
    active_learning.periodic = periodic
    active_learning.atom_style = atom_style
    with open(output_directory + "/simulation.lammps", "w") as f:
        f.write("run 5")

    with pytest.raises(ValueError) as e:
        active_learning._write_input_lammps(
            path=output_directory,
            seed=1,
            temperature=1,
            pressure=1.0,
            integrator=integrator,
        )

    assert str(e.value) == error


# TEST WRITE STRUCTURE LAMMPS


@pytest.mark.parametrize(
    "atom_style, output", [("atomic", [1, 1, 0, 0, 0]), ("full", [1, 0, 1, 0, 0, 0, 0])]
)
def test_write_structure_lammps(
    active_learning: ActiveLearning,
    atom_style: str,
    output: List[int],
):
    """
    Test the `_write_structure_lammps` function is successful for both `atom_style` values
    leading to differences in the output file.
    """
    output_directory = "tests/data/tests_output"
    active_learning.atom_style = atom_style
    with open(output_directory + "/simulation.lammps", "w") as f:
        f.write("run 5")

    frame = Frame(
        lattice=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        symbols=np.array(["H"]),
        charges=np.array([0]),
        positions=np.array([[0, 0, 0]]),
    )
    dataset = Dataset(frames=[frame], all_structures=active_learning.all_structures)

    active_learning._write_structure_lammps(
        path=output_directory,
        dataset=dataset,
        selection=0,
    )

    assert isfile(output_directory + "/structure.lammps")
    with open(output_directory + "/structure.lammps") as f:
        lines = f.readlines()
    assert np.all(np.array(lines[-1].split(), dtype=float) == np.array(output))


# TEST WRITE LAMMPS


@pytest.mark.parametrize("integrator", ["nve", "nvt", "npt"])
def test_write_lammps(active_learning: ActiveLearning, integrator: str):
    """
    Test the `write_lammps` function is successful, leading to the creation of empty
    joblist_mode1.dat for all integrators.
    """
    output_directory = "tests/data/tests_output"
    pressure = "_p1.0" if integrator == "npt" else ""
    with open(output_directory + "/simulation.lammps", "w") as f:
        f.write("run 5")
    active_learning.integrators = [integrator]

    active_learning.write_lammps(range(1))

    assert isfile(output_directory + "/joblist_mode1.dat")
    assert isfile(
        output_directory
        + "/mode1/test_{0}_hdnnp1_t0{1}_1/input.lammps".format(integrator, pressure)
    )
    assert isfile(
        output_directory
        + "/mode1/test_{0}_hdnnp2_t0{1}_2/input.lammps".format(integrator, pressure)
    )


def test_write_lammps_choose_weights(
    active_learning: ActiveLearning, capsys: pytest.CaptureFixture
):
    """
    Test the `write_lammps` function is successful when weights files are missing, and one
    is automatically selected.
    """
    # Create "simulation.lammps" to read from
    output_directory = "tests/data/tests_output"
    with open(output_directory + "/simulation.lammps", "w") as f:
        f.write("run 5")
    # Remove .data weights files, and create copies of the single "weights.001.000000.out"
    # file we have for each species at epoch 20 (this is the last epoch in learning-curve.out),
    # so is selected by default
    weights_files = [
        "tests/data/n2p2_copy/weights.001.data",
        "tests/data/n2p2_copy/weights.006.data",
        "tests/data/n2p2_copy/weights.008.data",
    ]
    for f in weights_files:
        remove(f)
        copy("tests/data/n2p2_copy/weights.001.000000.out", f[:-4] + "000020.out")

    active_learning.write_lammps(range(1))

    assert isfile(output_directory + "/joblist_mode1.dat")
    assert isfile(output_directory + "/mode1/test_npt_hdnnp1_t0_p1.0_1/input.lammps")
    assert isfile(output_directory + "/mode1/test_npt_hdnnp2_t0_p1.0_2/input.lammps")
    assert capsys.readouterr().out == (
        "Starting from the 0th structure every 1th structure of the input.data file is used.\n"
        "The given variations of the settings are repeated 1 times\n"
        "{0} not found, attempting to automatically choose one\n"
        "WARNING: The structures of the input.data file are used more than once.\n"
        "Input was generated for 2 simulations.\n"
        "".format(weights_files[0])
    )


def test_write_lammps_mode1_error(active_learning: ActiveLearning):
    """Test that an error is raised if the mode1 directory already exists."""
    mkdir("tests/data/tests_output/mode1")
    with pytest.raises(IOError) as e:
        active_learning.write_lammps(range(1))

    assert str(e.value) == (
        "Path mode1 already exists. Please remove old directory first if you would "
        "like to recreate it."
    )


@pytest.mark.parametrize("file", ["input.nn", "scaling.data"])
def test_write_lammps_missing_file_error(active_learning: ActiveLearning, file: str):
    """Tests that if scaling or input files are missing, then an Error is raised."""
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")
    remove("tests/data/n2p2_copy/" + file)

    with pytest.raises(OSError) as e:
        active_learning.write_lammps(range(1))

    assert str(e.value) == ("tests/data/n2p2_copy/{} not found".format(file))


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
    all_species: List[Species], active_learning: ActiveLearning
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


def test_write_lammps_selection(
    all_species: List[Species],
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
    all_species: List[Species], active_learning: ActiveLearning
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


# TEST READ LAMMPSTRJ


def test_read_lammpstrj(active_learning: ActiveLearning):
    """
    Test that `_read_lammpstrj` extracts information from "structure.lammpstrj", ignoring
    irrelevant lines, and returns the specified structure(s).
    """
    directory = "tests/data/tests_output"
    with open(
        "tests/data/active_learning/mode1/test_npt_hdnnp1_t325_p1.0_1/structure.lammpstrj"
    ) as f:
        text = f.read()
    with open(directory + "/structure.lammpstrj", "w") as f:
        f.write("not a timestep line\n")
        f.write(text)

    structures, structure_lines = active_learning._read_lammpstrj(
        timesteps=np.array([2]), directory=directory
    )

    assert len(structures) == 521
    assert structures[1] == "2\n"
    assert structure_lines == 521


def test_prepare_lammps_trajectory(active_learning: ActiveLearning):
    """Test that calling `prepare_lammps_trajectory` results in the extrapolation
    data being written to file."""
    # Generate the mode1 directory
    with open("tests/data/tests_output/simulation.lammps", "w") as f:
        f.write("run 5")
    prepare_mode1_files(files=["log.lammps", "structure.lammpstrj"])

    active_learning.prepare_lammps_trajectory()

    assert isfile(
        "tests/data/tests_output/mode1/test_npt_hdnnp1_t325_p1.0_1/extrapolation.dat"
    )


# TEST GET PATHS


def test_get_paths_os_error(active_learning: ActiveLearning):
    """
    Test that an Error is raised when attempting to `_get_paths` for an unknown
    `structure_name`.
    """
    structure_name = "missing"

    with pytest.raises(Exception) as e:
        active_learning._get_paths(structure_name=structure_name)

    assert str(e.value) == "Simulations with the structure name missing were not found."


def test_get_paths_value_error(active_learning: ActiveLearning):
    """
    Test that an Error is raised if the mode1 subdirectory contains no finished simulations.
    """
    mkdir("tests/data/tests_output/mode1")
    mkdir("tests/data/tests_output/mode1/nve_hdnnp")

    with pytest.raises(Exception) as e:
        active_learning._get_paths(structure_name="")

    assert str(e.value) == "None of the  simulations finished."


# TEST READ LOG FORMAT


@pytest.mark.parametrize(
    "text, expected_format",
    [
        ("**********\n\n   NNP LIBRARY v2.0.0", "v2.0.0"),
        ("**********\n\n\n\n\nn²p² version      : v2.1.1", "v2.1.1"),
        ("**********\n\n\n\n\nn²p² version  (from git): v2.1.4", "v2.1.1"),
    ],
)
def test_read_log_format(
    active_learning: ActiveLearning,
    text: str,
    expected_format: str,
):
    """
    Test that all expected log formats can be correctly determined from "log.lammps"
    """
    path = "directory"
    mkdir("tests/data/tests_output/mode1")
    mkdir("tests/data/tests_output/mode1/directory")
    with open("tests/data/tests_output/mode1/directory/log.lammps", "w") as f:
        f.write(text)

    extrapolation_format = active_learning._read_log_format(path)

    assert extrapolation_format == expected_format


@pytest.mark.parametrize(
    "text",
    [
        ("**********\n\n\n\n\nn²p² version      : v0.0.0"),
        ("\n\n\n\n\nn²p² version      : v2.1.1"),
    ],
)
def test_read_log_format_errors(active_learning: ActiveLearning, text: str):
    """
    Test that errors are raised for unrecognised "log.lammps" formats.
    """
    path = "directory"
    mkdir("tests/data/tests_output/mode1")
    mkdir("tests/data/tests_output/mode1/directory")
    with open("tests/data/tests_output/mode1/directory/log.lammps", "w") as f:
        f.write(text)

    with pytest.raises(OSError) as e:
        active_learning._read_log_format(path)

    assert str(e.value) == (
        "n2p2 extrapolation warning format cannot be identified in the file "
        "{0}/log.lammps. Known formats are corresponding to n2p2 v2.0.0 and v2.1.1."
        "".format(path)
    )


# TEST READ LOG


@pytest.mark.parametrize(
    "extrapolation_data, expected",
    [
        (np.array([-1, -1, -1, -1, -1]), ([-1, -1], [0, 0], [None, None])),
        (
            np.array([1, 1, 1, 521, 1]),
            ([1, -1], [1.0, 0.0], [[np.array([1.0, 486, 26])], [None]]),
        ),
    ],
)
@pytest.mark.parametrize(
    "extrapolation_format, text",
    [
        (
            "v2.0.0",
            (
                "thermo        0\n"
                "### NNP EXTRAPOLATION WARNING ### STRUCTURE:      0 ATOM:       486 "
                "SYMFUNC: 26 VALUE:  2.000E+00 MIN:  0.000E+00 MAX:  1.000E+00\n"
                "thermo        1     0.0000  374.000    -53015.39756    -52990.69413    "
                "4.7906   42.2281 -107549.23  18.58077  18.58077  18.58077  "
                "90.00000  90.00000  90.00000  0.89576\n"
                "final line is skipped"
            ),
        ),
        (
            "v2.1.1",
            (
                "thermo        0\n"
                "### NNP EXTRAPOLATION WARNING ### STRUCTURE:      0 ATOM:       486 "
                "ELEMENT:  C SYMFUNC:   26 TYPE:  2 VALUE:  2.000E+00 "
                "MIN:  0.000E+00 MAX:  1.000E+00\n"
                "thermo        1\n"
                "final line is skipped"
            ),
        ),
    ],
)
def test_read_log(
    active_learning: ActiveLearning,
    text: str,
    extrapolation_format: str,
    extrapolation_data: np.ndarray,
    expected: Tuple[List[int], List[float], List[List[List[Union[float, int]]]]],
):
    """
    Test that for all expected log formats, "log.lammps" can be correctly read and
    extrapolations match expected values.
    """
    path = "directory"
    mkdir("tests/data/tests_output/mode1")
    mkdir("tests/data/tests_output/mode1/directory")
    with open("tests/data/tests_output/mode1/directory/log.lammps", "w") as f:
        f.write(text)
    active_learning.tolerances = [0.5, 2.0]

    (
        extrapolation_timestep,
        extrapolation_value,
        extrapolation_statistic,
    ) = active_learning._read_log(
        path=path,
        extrapolation_data=extrapolation_data,
        extrapolation_format=extrapolation_format,
    )

    assert extrapolation_timestep == expected[0]
    assert extrapolation_value == expected[1]
    for i, value in enumerate(extrapolation_statistic):
        if isinstance(value, type(None)):
            assert value == expected[2][i]
        else:
            for j, inner_value in enumerate(value):
                if isinstance(inner_value, type(None)):
                    assert inner_value == expected[2][i][j]
                else:
                    np.testing.assert_allclose(inner_value, expected[2][i][j])


# TEST GET TIMESTEPS


@pytest.mark.parametrize(
    (
        "extrapolation_timesteps, extrapolation_values, extrapolation_data, tolerances, "
        "expected_selected_timesteps, expected_tolerance_indices, expected_smalls, stdout"
    ),
    [
        (
            np.array([[-1, -1]]),
            np.array([[0, 0]]),
            [np.array([-1, -1, -1, -1, -1])],
            [0.01, 0.1],
            [[-1, []]],
            [-1],
            np.array([0]),
            (
                "Only less than 0.1% of the simulations show an extrapolation if a tolerance "
                "of {0} is employed (the initial {1} time steps are neglected). The tolerance "
                "value is reduced to {2}.\n"
                "There are no small extrapolations.\n"
                "WARNING: A simulation ended due to too many extrapolations but no one of "
                "these was larger than the tolerance of {3}. If this message is printed "
                "several times you should consider to reduce the first and second entry of "
                "tolerances.\n".format(0.1, 20, 0.01, 0.01)
            ),
        ),
        (
            np.array([[101, 102, 103, 104]]),
            np.array([[0.1, 0.2, 0.3, 0.4]]),
            [np.array([-1, -1, -1, -1, 104])],
            [0.05, 0.1, 0.15, 0.2],
            [[102, np.array([103, 104])]],
            [-1],
            np.array([1]),
            ("Small extrapolations are present in 1 of 1 simulations (100.0%).\n"),
        ),
        (
            np.array([[101, 102, 103, 104]]),
            np.array([[np.nan, np.nan, np.nan, np.nan]]),
            [np.array([-1, -1, -1, -1, 104])],
            [0.05, 0.1, 0.15, 0.2],
            [[102, np.array([103, 104])]],
            [-1],
            np.array([1]),
            ("Small extrapolations are present in 1 of 1 simulations (100.0%).\n"),
        ),
        (
            np.array([[101, 102, -1, -1]]),
            np.array([[0.06, 0.11, 0.0, 0.0]]),
            [np.array([-1, -1, -1, -1, 104])],
            [0.05, 0.1, 0.15, 0.2],
            [[102, np.array([-1, -1])]],
            [0],
            np.array([1]),
            (
                "Small extrapolations are present in 1 of 1 simulations (100.0%).\n"
                "There are no large extrapolations.\n"
            ),
        ),
        (
            np.array([[101, -1, -1, -1], [101, 102, -1, -1]]),
            np.array([[0.06, 0.09, 0.0, 0.0], [0.06, 0.11, 0.0, 0.0]]),
            [np.array([-1, -1, -1, -1, 104]), np.array([-1, -1, -1, -1, 104])],
            [0.05, 0.1, 0.15, 0.2],
            [[101, np.array([-1, -1])], [102, np.array([-1, -1])]],
            [-1, 0],
            np.array([0, 1]),
            (
                "Small extrapolations are present in 1 of 2 simulations (50.0%).\n"
                "There are no large extrapolations.\n"
                "WARNING: A simulation ended due to too many extrapolations but no one of "
                "these was larger than the tolerance of 0.1. If this message is printed "
                "several times you should consider to reduce the first and second entry of "
                "tolerances.\n"
                "With the reduction of the tolerance to 0.05 an extrapolated structure could "
                "be found in this case.\n"
            ),
        ),
        (
            np.array([[50001, 50002, -1, -1]]),
            np.array([[0.06, 0.11, 0.0, 0.0]]),
            [np.array([-1, -1, -1, -1, 100000])],
            [0.05, 0.1, 0.15, 0.2],
            [[20000, 30000, 40000, 50002, np.array([-1, -1])]],
            [0],
            np.array([1]),
            (
                "Small extrapolations are present in 1 of 1 simulations (100.0%).\n"
                "There are no large extrapolations.\n"
            ),
        ),
        (
            np.array([[10001, 10002, -1, -1]]),
            np.array([[0.06, 0.11, 0.0, 0.0]]),
            [np.array([-1, -1, -1, -1, 100000])],
            [0.05, 0.1, 0.15, 0.2],
            [[4000, 6000, 8000, 10002, np.array([-1, -1])]],
            [0],
            np.array([1]),
            (
                "Small extrapolations are present in 1 of 1 simulations (100.0%).\n"
                "There are no large extrapolations.\n"
            ),
        ),
        (
            np.array([[-1, -1, -1, -1]]),
            np.array([[0.06, 0.11, 0.0, 0.0]]),
            [np.array([-1, -1, -1, -1, 100000])],
            [0.05, 0.1, 0.15, 0.2],
            [
                [
                    20000,
                    30000,
                    40000,
                    50000,
                    60000,
                    70000,
                    80000,
                    90000,
                    100000,
                    -1,
                    np.array([-1, -1]),
                ]
            ],
            [-1],
            np.array([0]),
            (
                "Only less than 0.1% of the simulations show an extrapolation if a tolerance "
                "of 0.1 is employed (the initial 20 time steps are neglected). The tolerance "
                "value is reduced to 0.05.\n"
                "There are no small extrapolations.\n"
                "WARNING: A simulation ended due to too many extrapolations but no one of "
                "these was larger than the tolerance of 0.05. If this message is printed "
                "several times you should consider to reduce the first and second entry of "
                "tolerances.\n"
            ),
        ),
        (
            np.array([[-1, -1, -1, -1]]),
            np.array([[0.06, 0.11, 0.0, 0.0]]),
            [np.array([-1, -1, -1, -1, 50000])],
            [0.05, 0.1, 0.15, 0.2],
            [[20000, 30000, 40000, 50000, -1, np.array([-1, -1])]],
            [-1],
            np.array([0]),
            (
                "Only less than 0.1% of the simulations show an extrapolation if a tolerance "
                "of 0.1 is employed (the initial 20 time steps are neglected). The tolerance "
                "value is reduced to 0.05.\n"
                "There are no small extrapolations.\n"
                "WARNING: A simulation ended due to too many extrapolations but no one of "
                "these was larger than the tolerance of 0.05. If this message is printed "
                "several times you should consider to reduce the first and second entry of "
                "tolerances.\n"
                "Included the last regularly dumped structure of the simulation as it ended "
                "due to too many extrapolations.\n"
            ),
        ),
        (
            np.array([[-1, -1, -1, -1]]),
            np.array([[0.06, 0.11, 0.0, 0.0]]),
            [np.array([-1, -1, -1, -1, 300])],
            [0.05, 0.1, 0.15, 0.2],
            [[300, -1, np.array([-1, -1])]],
            [-1],
            np.array([0]),
            (
                "Only less than 0.1% of the simulations show an extrapolation if a tolerance "
                "of 0.1 is employed (the initial 20 time steps are neglected). The tolerance "
                "value is reduced to 0.05.\n"
                "There are no small extrapolations.\n"
                "WARNING: A simulation ended due to too many extrapolations but no one of "
                "these was larger than the tolerance of 0.05. If this message is printed "
                "several times you should consider to reduce the first and second entry of "
                "tolerances.\n"
                "Included the last regularly dumped structure of the simulation as it ended "
                "due to too many extrapolations.\n"
            ),
        ),
    ],
)
def test_get_timesteps(
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
    extrapolation_timesteps: np.ndarray,
    extrapolation_values: np.ndarray,
    extrapolation_data: List[np.ndarray],
    tolerances: List[int],
    expected_selected_timesteps: List[List[Union[int, List[int]]]],
    expected_tolerance_indices: List[int],
    expected_smalls: np.ndarray,
    stdout: str,
):
    """
    Test that `get_timesteps` returns the expected outcome(s) and prints the expected
    information for a variety of initial conditions.
    """
    active_learning.initial_tolerance = 2
    active_learning.tolerances = tolerances
    structure = list(active_learning.all_structures.values())[0]

    (
        selected_timesteps,
        tolerance_indices,
        smalls,
        n_small,
    ) = active_learning._get_timesteps(
        extrapolation_timesteps=extrapolation_timesteps,
        extrapolation_values=extrapolation_values,
        extrapolation_data=extrapolation_data,
        structure=structure,
    )

    assert len(selected_timesteps) == len(expected_selected_timesteps)
    assert len(tolerance_indices) == len(expected_tolerance_indices)
    assert len(smalls) == len(expected_smalls)
    for i in range(len(selected_timesteps)):
        assert len(selected_timesteps[i]) == len(expected_selected_timesteps[i])
        for j in range(len(selected_timesteps[i]) - 1):
            assert selected_timesteps[i][j] == expected_selected_timesteps[i][j]

        assert len(selected_timesteps[i][-1]) == len(expected_selected_timesteps[i][-1])
        if len(selected_timesteps[i][-1]) > 1:
            assert all(selected_timesteps[i][-1] == expected_selected_timesteps[i][-1])
        else:
            assert selected_timesteps[i][-1] == expected_selected_timesteps[i][-1]

        assert tolerance_indices[i] == expected_tolerance_indices[i]
        assert smalls[i] == expected_smalls[i]

    assert n_small == 2
    assert capsys.readouterr().out == stdout


def test_prepare_data_new(
    active_learning: ActiveLearning, capsys: pytest.CaptureFixture
):
    """
    Test that creating "input.data-new" results in the expected stdout, and that if
    called again then a warning is printed that the file is in memory or that the
    existing file will be re-used.
    """
    prepare_mode1_files()

    active_learning.prepare_data_new()

    assert capsys.readouterr().out == PREPARE_DATA_NEW_STDOUT

    active_learning.prepare_data_new()

    assert capsys.readouterr().out == (
        "`self.dataset_new` or `self.names` is already set. "
        "To regenerate the new dataset, set them to None.\n"
    )

    active_learning.dataset_new = Dataset()
    active_learning.names = []
    active_learning.prepare_data_new()

    assert capsys.readouterr().out == (
        "input.data-new is already present and data will be read from there.\n"
        "To regenerate input.data-new, delete existing file.\n"
    )

    active_learning.dataset_new = None
    active_learning.names = None
    active_learning.prepare_data_new()

    assert capsys.readouterr().out == (
        "input.data-new is already present and data will be read from there.\n"
        "To regenerate input.data-new, delete existing file.\n"
    )


def test_prepare_data_new_name_none(
    controller: Controller, all_species: List[Species], capsys: pytest.CaptureFixture
):
    """Test that `prepare_data_new` does not print the name of the Structure in the case
    where we only have one without a defined name."""
    prepare_mode1_files()
    copy("tests/data/scripts/template.sh", "tests/data/tests_output/template.sh")
    structure = Structure(name=None, all_species=all_species, delta_E=1.0, delta_F=1.0)
    controller.all_structures = AllStructures(structure)
    active_learning = ActiveLearning(
        data_controller=controller,
    )

    active_learning.prepare_data_new()

    assert "Structure: " not in capsys.readouterr().out


def test_prepare_data_add(
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
    basic_dataset: Dataset,
):
    """Test that calling `prepare_data_add` gives the expected stdout,
    including values of timings and numbers of structures."""
    prepare_mode2_files()

    active_learning.names = np.array(["test_s0"])
    active_learning.dataset_new = basic_dataset
    active_learning.prepare_data_add()

    assert (
        capsys.readouterr().out
        == "\nTime to calculate 1 structures using RuNNer: "
        + "HDNNP_1: 1.02 h, HDNNP_2: 1.02 min.\n\n"
        + "0 structures identified over energy threshold `dE=[1.]`:\n"
        + "[]\n"
        + "1 structures identified over force threshold `dF=[1.]`:\n"
        + "[0.]\n"
        + "1 missing structures were identified.\n"
        + "1 missing structures originate from small extrapolations.\n"
        + "0 missing structures originate from large extrapolations.\n"
    )


def test_prepare_data_add_call_prepare_data_new(
    active_learning: ActiveLearning, capsys: pytest.CaptureFixture
):
    """Test that in the case that `active_learning.positions` etc. are not set,
    then `prepare_data_new` is called (evidenced by its stdout). In the case where the
    variables are not set but the "input.data-new" is present, it should be read instead."""
    prepare_mode1_files()
    prepare_mode2_files()

    active_learning.prepare_data_add()

    assert active_learning.selection == [0]
    assert active_learning.names == np.array(["test_npt_hdnnp1_t325_p1.0_1_s23"])
    assert (
        capsys.readouterr().out
        == "No data loaded, calling `prepare_data_new`\n"
        + PREPARE_DATA_NEW_STDOUT
        + "\nTime to calculate 1 structures using RuNNer: "
        + "HDNNP_1: 1.02 h, HDNNP_2: 1.02 min.\n\n"
        + "0 structures identified over energy threshold `dE=[1.]`:\n"
        + "[]\n"
        + "1 structures identified over force threshold `dF=[1.]`:\n"
        + "[0.]\n"
        + "1 missing structures were identified.\n"
        + "0 missing structures originate from small extrapolations.\n"
        + "1 missing structures originate from large extrapolations.\n"
    )

    active_learning.names = []
    active_learning.dataset_new = Dataset(all_structures=active_learning.all_structures)

    active_learning.prepare_data_add()

    assert active_learning.selection == [0]
    assert active_learning.names == np.array(["test_npt_hdnnp1_t325_p1.0_1_s23"])
    assert (
        capsys.readouterr().out
        == "No data loaded, calling `prepare_data_new`\n"
        + "input.data-new is already present and data will be read from there.\n"
        + "To regenerate input.data-new, delete existing file.\n"
        + "\nTime to calculate 1 structures using RuNNer: "
        + "HDNNP_1: 1.02 h, HDNNP_2: 1.02 min.\n\n"
        + "0 structures identified over energy threshold `dE=[1.]`:\n"
        + "[]\n"
        + "1 structures identified over force threshold `dF=[1.]`:\n"
        + "[0.]\n"
        + "1 missing structures were identified.\n"
        + "0 missing structures originate from small extrapolations.\n"
        + "1 missing structures originate from large extrapolations.\n"
    )


def test_prepare_data_add_normalisation(
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
    basic_dataset: Dataset,
):
    """Test that if normalisation is present in the N2P2 headers,
    then this is applied to the energy and force threshold values."""
    prepare_mode2_files()

    active_learning.names = np.array(["test_s0"])
    active_learning.dataset_new = basic_dataset
    copy("tests/data/n2p2_copy/input.nn.norm", "tests/data/n2p2_copy/input.nn")

    active_learning.prepare_data_add()

    assert (
        capsys.readouterr().out
        == "\nTime to calculate 1 structures using RuNNer: "
        + "HDNNP_1: 1.02 h, HDNNP_2: 1.02 min.\n\n"
        + "`dE` and `dF` converted to internal (normalised) network units: "
        + "`dE=[1375.87882603]`, `dF=[43.41918089]`\n"
        + "0 structures identified over energy threshold `dE=[1375.87882603]`:\n"
        + "[]\n"
        + "0 structures identified over force threshold `dF=[43.41918089]`:\n"
        + "[]\n"
        + "1 missing structures were identified.\n"
        + "1 missing structures originate from small extrapolations.\n"
        + "0 missing structures originate from large extrapolations.\n"
    )


def test_prepare_data_add_mixed_normalisation(
    active_learning: ActiveLearning, basic_dataset: Dataset
):
    """Test that a ValueError is raised if the two "input.nn" files
    have different normalisation headers."""
    prepare_mode2_files()

    active_learning.names = np.array(["test_s0"])
    active_learning.data_controller.n2p2_directories = [
        "tests/data/n2p2_copy",
        "tests/data/n2p2",
    ]
    active_learning.dataset_new = basic_dataset
    copy("tests/data/n2p2_copy/input.nn.norm", "tests/data/n2p2_copy/input.nn")

    with pytest.raises(ValueError) as e:
        active_learning.prepare_data_add()

    assert str(e.value) == (
        "Normalisation factors conv_energy=1375.8788260330357, conv_length=31.688272275479928 "
        "in tests/data/n2p2_copy are different to conv_energy=1.0, conv_length=1.0 "
        "in tests/data/n2p2"
    )


def test_prepare_data_add_error(active_learning: ActiveLearning):
    """
    Test that an error is raised when calling `prepare_data_add` without a `mode2` directory.
    """

    with pytest.raises(OSError) as e:
        active_learning.prepare_data_add()

    assert str(e.value) == "`mode2` directory not found."


def test_print_statistics_no_selection(
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
    basic_dataset: Dataset,
):
    """Test that if we do not have any statistics,
    then no information relating to them is printed."""
    prepare_mode2_files()
    selection = np.array([0])
    names = np.array(["test_s0"])

    basic_dataset[0].statistics = []
    active_learning.dataset_new = basic_dataset
    active_learning._print_statistics(selection=selection, names=names)

    assert capsys.readouterr().out == "1 missing structures were identified.\n"


def test_print_statistics_multiple_names(
    all_species: List[Species],
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
    basic_dataset: Dataset,
):
    """Test that if multiple Structures are present then we separate the information
    with headings when printing."""
    prepare_mode2_files()

    structure_0 = Structure(
        name="test0", all_species=all_species, delta_E=1.0, delta_F=1.0
    )
    structure_1 = Structure(
        name="test1", all_species=all_species, delta_E=1.0, delta_F=1.0
    )
    active_learning.all_structures = AllStructures(structure_0, structure_1)
    selection = np.array([0])
    names = np.array(["test0_s0", "test1_s0"])

    basic_dataset.append(basic_dataset[0])
    basic_dataset[0].name = "test0"
    basic_dataset[1].name = "test1"
    active_learning.dataset_new = basic_dataset
    active_learning._print_statistics(selection=selection, names=names)

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
    all_species: List[Species], active_learning: ActiveLearning, basic_dataset: Dataset
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
    selection = np.array([0])
    names = np.array(["test_s0"])
    active_learning.all_structures = all_structures

    active_learning.dataset_new = basic_dataset
    selection = active_learning._improve_selection(
        selection=selection,
        names=names,
    )

    assert selection == np.array([0])


def test_improve_selection_all_empty_statistics(
    all_species: List[Species], active_learning: ActiveLearning, basic_dataset: Dataset
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
    selection = np.array([0, 1])
    names = np.array(["test_s0", "test_s1"])
    active_learning.all_structures = all_structures

    basic_dataset[0].statistics = []
    basic_dataset.append(basic_dataset[0])
    active_learning.dataset_new = basic_dataset
    selection = active_learning._improve_selection(
        selection=selection,
        names=names,
    )

    assert all(selection == np.array([0, 1]))


def test_improve_selection_reduce_selection(
    all_species: List[Species], active_learning: ActiveLearning, basic_dataset: Dataset
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
    ordered_structure_names = [
        "test0",
        "test0",
        "test0",
        "test1",
        "test1",
        "test1",
        "test1",
    ]
    active_learning.all_structures = all_structures
    structures = list(all_structures.values())
    active_learning._validate_timesteps(
        active_learning.timestep, active_learning.N_steps, structures
    )

    basic_dataset[0].name = ordered_structure_names[0]
    for name in ordered_structure_names[1:]:
        basic_dataset.append(basic_dataset[0])
        basic_dataset[-1].name = name

    active_learning.dataset_new = basic_dataset
    selection = active_learning._improve_selection(
        selection=selection,
        names=names,
    )

    assert all(selection == np.array([0, 2, 3, 5]))


def test_improve_selection_max_extrapolated_structures(
    all_species: List[Species],
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
    basic_dataset: Dataset,
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
    selection = np.array([0])
    names = np.array(["test_s0", "test_s0"])
    active_learning.all_structures = all_structures
    structures = list(all_structures.values())
    active_learning._validate_timesteps(
        active_learning.timestep, active_learning.N_steps, structures
    )

    basic_dataset.append(basic_dataset[0])
    active_learning.dataset_new = basic_dataset
    selection = active_learning._improve_selection(
        selection=selection,
        names=names,
    )

    assert selection == np.array([0]) or selection == np.array([1])
    assert capsys.readouterr().out == "The extrapolation ['H', '0'] occurred 2 times.\n"


def test_improve_selection_exceptions(
    all_species: List[Species], active_learning: ActiveLearning, basic_dataset: Dataset
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
    selection = np.array([0])
    names = np.array(["test_s0"])
    active_learning.all_structures = all_structures
    structures = list(all_structures.values())
    active_learning._validate_timesteps(
        active_learning.timestep, active_learning.N_steps, structures
    )

    basic_dataset.append(basic_dataset[0])
    active_learning.dataset_new = basic_dataset
    selection = active_learning._improve_selection(
        selection=selection,
        names=names,
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


# TEST GET STRUCTURE


def test_get_structure(
    active_learning: ActiveLearning,
):
    """Test that all properties are correctly extracted from the data."""
    data = [
        "ITEM: TIMESTEP",
        "0",
        "ITEM: NUMBER OF ATOMS",
        "1",
        "ITEM: BOX BOUNDS xy xz yz",
        "0.0000000000000000e+00 1. 0.0000000000000000e+00",
        "0.0000000000000000e+00 1. 0.0000000000000000e+00",
        "0.0000000000000000e+00 1. 0.0000000000000000e+00",
        "ITEM: ATOMS id element x y z q",
        "1 O 12.3466 3.95684 5.12344 0.000",
    ]

    frame = active_learning._get_structure(
        data=data,
    )

    assert np.allclose(
        frame.lattice, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert frame.symbols == np.array("O")
    assert np.allclose(frame.positions, np.array([[12.3466, 3.95684, 5.12344]]))
    assert frame.charges == np.array([0.0])


# TEST READ STRUCTURES


@pytest.mark.parametrize(
    "selected_timestep, expected_index",
    [
        ([0, 1, [2, 2, 2]], -1),
        ([100, 101, [102, 102, 102]], 1),
        ([-1, 100, [101, 101, 101]], 1),
        ([-1, -1, [100, 100, 100]], 1),
    ],
)
def test_read_structures(
    active_learning: ActiveLearning,
    selected_timestep: List[int],
    expected_index: List[int],
):
    """
    Test that for a variety of different `selected_timestep`s, the correct index is returned.
    """
    path = "test_npt_hdnnp1_t325_p1.0_1"
    extrapolation_data = np.array([-1, -1, -1, 521, -1])
    n_small = 2
    small = 1
    tolerance_index = 1
    extrapolation_statistic = [
        [[1.0, 1, 1]],
        [[1.0, 1, 1]],
        [[1.0, 1, 1]],
        [[1.0, 1, 1]],
        [[1.0, 1, 1]],
    ]
    element2index = {"H": 0, "C": 1, "O": 2}
    structure = active_learning.all_structures["test"]
    copytree(
        "tests/data/active_learning/mode1",
        "tests/data/tests_output/mode1",
    )

    returned_index = active_learning._read_structures(
        path=path,
        extrapolation_data=extrapolation_data,
        selected_timestep=selected_timestep,
        n_small=n_small,
        small=small,
        tolerance_index=tolerance_index,
        extrapolation_statistic=extrapolation_statistic,
        element2index=element2index,
        structure=structure,
    )

    assert returned_index == expected_index


@pytest.mark.parametrize(
    "extrapolation_timesteps, tolerance_indices, output",
    [
        (
            np.array([[1, 3], [2, 4]]),
            np.array([1, 1]),
            (
                "The median number of time steps to a small extrapolation is "
                "{0} (HDNNP_1: {1}, HDNNP_2: {2}).\n"
                "The median number of time steps between the first and second selected "
                "extrapolated structure is {3} (HDNNP_1: {4}, HDNNP_2: {5}).\n"
                "".format(2, 1, 2, 2, 2, 2)
            ),
        ),
        (
            np.array([[-1, -1], [2, 4]]),
            np.array([-1, 1]),
            (
                "The median number of time steps to a small extrapolation is "
                "{0} (HDNNP_1: {1}, HDNNP_2: {2}).\n"
                "The median number of time steps between the first and second selected "
                "extrapolated structure is {3} (HDNNP_1: {4}, HDNNP_2: {5}).\n"
                "".format(2, "nan", 2, 2, "nan", 2)
            ),
        ),
        (
            np.array([[1, 3], [-1, -1]]),
            np.array([1, -1]),
            (
                "The median number of time steps to a small extrapolation is "
                "{0} (HDNNP_1: {1}, HDNNP_2: {2}).\n"
                "The median number of time steps between the first and second selected "
                "extrapolated structure is {3} (HDNNP_1: {4}, HDNNP_2: {5}).\n"
                "".format(1, 1, "nan", 2, 2, "nan")
            ),
        ),
        (np.array([[-1, -1], [-1, -1]]), np.array([-1, -1]), ""),
    ],
)
def test_print_reliability(
    active_learning: ActiveLearning,
    capsys: pytest.CaptureFixture,
    extrapolation_timesteps: np.ndarray,
    tolerance_indices: np.ndarray,
    output: str,
):
    """
    Test that `_print_reliability` extracts all information correctly.
    """
    active_learning._print_reliability(
        extrapolation_timesteps=extrapolation_timesteps,
        smalls=np.array([0, 0]),
        tolerance_indices=tolerance_indices,
        paths=np.array(["_hdnnp1_", "_hdnnp2_"]),
    )

    assert capsys.readouterr().out == output
