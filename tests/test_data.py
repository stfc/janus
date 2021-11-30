"""
Unit tests for `data.py`
"""

from os import listdir, mkdir, remove
from os.path import isfile, join
from shutil import copy, rmtree
from typing import List

from ase.atoms import Atoms
from genericpath import isdir
import numpy as np
import pytest

from cc_hdnnp.data import Data
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


def test_data_read_trajectory(data: Data):
    """
    Test that a trajectory can be read from file successfully.
    """
    data.read_trajectory("trajectory.history")

    assert len(data.trajectory) == 2
    assert data.trajectory[0].cell[0, 0] == 17.7128441229


def test_data_read_trajectory_bohr(data: Data):
    """
    Test that a trajectory can be read from file successfully, with length unit conversion.
    """
    data.read_trajectory("trajectory.history", unit_in="Bohr")

    assert len(data.trajectory) == 2
    assert data.trajectory[0].cell[0, 0] == 9.373233444108351


@pytest.mark.parametrize(
    "file_xyz, single_output",
    [
        ("tests_output/{}.xyz", False),
        ("tests_output/sub_directory/{}.xyz", False),
        ("tests_output/file.xyz", True),
        ("tests_output/sub_directory/file.xyz", True),
    ],
)
def test_data_convert_active_learning_to_xyz(
    data: Data, file_xyz: str, single_output: bool
):
    """
    Test that active learning structures can be written in xyz format, creating a subdirectory
    if needed, and a trailing newline.
    """
    data.convert_active_learning_to_xyz(
        file_n2p2_data="input.data-add",
        file_xyz=file_xyz,
        single_output=single_output,
    )

    file_out = "tests/data/" + file_xyz.format(0)
    assert isfile(file_out)
    with open(file_out) as f:
        text = f.read()
    assert text.endswith("\n")


def test_data_write_xyz(data: Data):
    """
    Test that a trajectory can be wtitten to xyz files successfully.
    """
    data.read_trajectory("trajectory.history")
    data.write_xyz(file_xyz="tests_output/{}.xyz")

    assert isfile("tests/data/tests_output/0.xyz")
    assert isfile("tests/data/tests_output/1.xyz")

    with open("tests/data/tests_output/0.xyz") as f:
        lines = f.readlines()
        lattice = lines[1].split('"')[1]
        position = lines[2].split()[1]
        assert lattice.split()[0] == "17.7128441229"
        assert position == "15.40516331"


def test_data_write_xyz_bohr(data: Data):
    """
    Test that a trajectory can be wtitten to xyz files successfully with Bohr units.
    """
    data.read_trajectory("trajectory.history")
    data.write_xyz(file_xyz="tests_output/{}.xyz", unit_out="Bohr")

    assert isfile("tests/data/tests_output/0.xyz")
    assert isfile("tests/data/tests_output/1.xyz")

    with open("tests/data/tests_output/0.xyz") as f:
        lines = f.readlines()
        lattice = lines[1].split('"')[1]
        position = lines[2].split()[1]
        assert lattice.split()[0] == "33.47242430192122"
        assert position == "29.11153958"


def test_scale_xyz(data: Data):
    """
    Test that xyz files can be scaled successfully.
    """
    base_cell_length = 17.7128441229
    base_position = 15.40516331
    scale_factor = 0.05
    data.scale_xyz(
        file_xyz_in="cp2k_input/{}.xyz",
        file_xyz_out="tests_output/{}.xyz",
        n_config=1,
        scale_factor=scale_factor,
    )

    assert isfile("tests/data/tests_output/0.xyz")

    with open("tests/data/tests_output/0.xyz") as f:
        lines = f.readlines()
        lattice = lines[1].split('"')[1]
        cell_length = float(lattice.split()[0])
        position = float(lines[2].split()[1])
        assert cell_length == pytest.approx((1 - scale_factor) * base_cell_length)
        assert position == pytest.approx((1 - scale_factor) * base_position)


def test_scale_xyz_random(data: Data):
    """
    Test that xyz files can be randomly scaled successfully.
    """
    base_cell_length = 17.7128441229
    base_position = 15.40516331
    scale_factor = 0.05
    data.scale_xyz(
        file_xyz_in="cp2k_input/{}.xyz",
        file_xyz_out="tests_output/{}.xyz",
        n_config=1,
        scale_factor=scale_factor,
        randomise=True,
    )

    assert isfile("tests/data/tests_output/0.xyz")

    with open("tests/data/tests_output/0.xyz") as f:
        lines = f.readlines()
        lattice = lines[1].split('"')[1]
        cell_length = float(lattice.split()[0])
        position = float(lines[2].split()[1])
        assert cell_length != base_cell_length
        assert cell_length < (1 + scale_factor) * base_cell_length
        assert cell_length > (1 - scale_factor) * base_cell_length
        assert position != base_position
        assert position < (1 + scale_factor) * base_position
        assert position > (1 - scale_factor) * base_position


def test_write_cp2k(data: Data):
    """
    Test that cp2k input and batch scripts are written to file.
    """
    # Copy template to the output directory to prevent removing them during clearup.
    copy("tests/data/cp2k_input/template.inp", "tests/data/tests_output/template.inp")
    copy("tests/data/cp2k_input/template.sh", "tests/data/tests_output/template.sh")

    data.scripts_directory = "tests/data/tests_output"
    data.write_cp2k(
        structure_name="test",
        basis_set="test",
        potential="test",
        file_bash="all.sh",
        file_batch="{}.sh",
        file_input="tests_output/{}.inp",
        file_xyz="cp2k_input/{}.xyz",
        n_config=1,
    )

    assert isfile("tests/data/tests_output/all.sh")
    assert isfile("tests/data/tests_output/n_0.sh")
    assert isfile("tests/data/tests_output/n_0.inp")


def test_write_cp2k_kwargs(data: Data):
    """
    Test that cp2k input and batch scripts are written to file with n_config, cutoff and
    relcutoff given.
    """
    # Copy template to the output directory to prevent removing them during clearup.
    copy("tests/data/cp2k_input/template.inp", "tests/data/tests_output/template.inp")
    copy("tests/data/cp2k_input/template.sh", "tests/data/tests_output/template.sh")

    data.scripts_directory = "tests/data/tests_output"
    data.write_cp2k(
        structure_name="test",
        basis_set="test",
        potential="test",
        file_bash="all.sh",
        file_batch="{}.sh",
        file_input="tests_output/{}.inp",
        file_xyz="cp2k_input/{}.xyz",
        n_config=1,
        cutoff=(60.0,),
        relcutoff=(600.0,),
    )

    assert isfile("tests/data/tests_output/all.sh")
    assert isfile("tests/data/tests_output/n_0_cutoff_60.0_relcutoff_600.0.sh")
    assert isfile("tests/data/tests_output/n_0_cutoff_60.0_relcutoff_600.0.inp")


def test_write_cp2k_kwargs_floats(data: Data):
    """
    Test that cp2k input and batch scripts are written to file with n_config, cutoff and
    relcutoff given as floats.
    """
    # Copy template to the output directory to prevent removing them during clearup.
    copy("tests/data/cp2k_input/template.inp", "tests/data/tests_output/template.inp")
    copy("tests/data/cp2k_input/template.sh", "tests/data/tests_output/template.sh")

    data.scripts_directory = "tests/data/tests_output"
    data.write_cp2k(
        structure_name="test",
        basis_set="test",
        potential="test",
        file_bash="all.sh",
        file_batch="{}.sh",
        file_input="tests_output/{}.inp",
        file_xyz="cp2k_input/{}.xyz",
        n_config=1,
        cutoff=60.0,
        relcutoff=600.0,
    )

    assert isfile("tests/data/tests_output/all.sh")
    assert isfile("tests/data/tests_output/n_0_cutoff_60.0_relcutoff_600.0.sh")
    assert isfile("tests/data/tests_output/n_0_cutoff_60.0_relcutoff_600.0.inp")


def test_data_print_cp2k_table(data: Data, capsys: pytest.CaptureFixture):
    """
    Test that cp2k output summary table is printed correctly.
    """
    data.print_cp2k_table(
        file_output="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        n_config=1,
    )

    assert (
        capsys.readouterr().out == "| Processes | Energy                | t/step (s) "
        "| t total (s) | Grid 1 | Grid 2 | Grid 3 | Grid 4 |\n"
        "|         1 | -1949.541521178710354 | 4.8 "
        "|    324.338 | 299035 | 219890 | 286470 | 252366 |\n"
    )


def test_data_print_cp2k_table_kwargs(data: Data, capsys: pytest.CaptureFixture):
    """
    Test that cp2k output summary table is printed correctly with **kwargs.
    """
    data.print_cp2k_table(
        file_output="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        n_config=1,
        cutoff=(60.0,),
        relcutoff=(600.0,),
    )

    assert (
        capsys.readouterr().out
        == "| Cutoff |  Relative Cutoff | Processes | Energy                | t/step (s) "
        "| t total (s) | Grid 1 | Grid 2 | Grid 3 | Grid 4 |\n"
        "|   60.0 |   600.0 |         1 | -1949.541521178710354 | 4.8 "
        "|    324.338 | 299035 | 219890 | 286470 | 252366 |\n"
    )


def test_data_print_cp2k_table_no_total_time(data: Data, capsys: pytest.CaptureFixture):
    """
    Test that cp2k output summary table is printed correctly when total_time cannot be set.
    """
    data.print_cp2k_table(file_output="cp2k_output/no_timings_energy.log", n_config=1)

    assert (
        capsys.readouterr().out == "| Processes | Energy                | t/step (s) "
        "| t total (s) | Grid 1 | Grid 2 | Grid 3 | Grid 4 |\n"
        "|         1 | None | None |    None | 299035 | 219890 | 286470 | 252366 |\n"
    )


def test_data_write_n2p2_data(data: Data):
    """
    Test that n2p2 data is written to file successfully.
    """
    data.n2p2_directories = ["tests/data/tests_output"]
    data.write_n2p2_data(
        structure_name="test",
        file_cp2k_out="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
        file_xyz="cp2k_input/{}.xyz",
        file_n2p2_input="input.data",
        n_config=1,
    )

    assert isfile("tests/data/tests_output/input.data")


def test_data_write_n2p2_data_wrong_name(data: Data):
    """
    Test that an error is raised when the incorrect name is provided.
    """
    structure_name = "not_test"
    with pytest.raises(ValueError) as e:
        data.write_n2p2_data(
            structure_name=structure_name,
            file_cp2k_out="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
            file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
            file_xyz="cp2k_input/{}.xyz",
            file_n2p2_input="tests_output/input.data",
            n_config=1,
        )

    assert str(e.value) == "`structure_name` {} not recognized".format(structure_name)


def test_data_write_n2p2_data_no_energy(data: Data):
    """
    Test that an error is raised when energy missing from the log file.
    """
    file_cp2k_out = "cp2k_output/no_timings_energy.log"
    with pytest.raises(ValueError) as e:
        data.write_n2p2_data(
            structure_name="test",
            file_cp2k_out=file_cp2k_out,
            file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
            file_xyz="cp2k_input/{}.xyz",
            file_n2p2_input="tests_output/input.data",
            n_config=1,
        )

    assert str(e.value) == "Energy not found in {}".format(file_cp2k_out)


def test_data_write_n2p2_data_units(data: Data):
    """
    Test that n2p2 data is written to file successfully with units provided.
    """
    data.n2p2_directories = ["tests/data/tests_output"]
    data.write_n2p2_data(
        structure_name="test",
        file_cp2k_out="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
        file_xyz="cp2k_input/{}.xyz",
        file_n2p2_input="input.data",
        n_config=1,
        n2p2_units={"length": "Ang", "energy": "eV", "force": "eV / Ang"},
    )

    assert isfile("tests/data/tests_output/input.data")


def test_data_write_n2p2_data_appended(data: Data):
    """
    Test that n2p2 data is appended to file without overwrite if output file already exists.
    """
    data.n2p2_directories = ["tests/data/tests_output"]
    with open("tests/data/tests_output/input.data", "w") as f:
        f.write("test text\n")

    data.write_n2p2_data(
        structure_name="test",
        file_cp2k_out="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
        file_xyz="cp2k_input/{}.xyz",
        file_n2p2_input="input.data",
        n_config=1,
        n2p2_units={"length": "Ang", "energy": "eV", "force": "eV / Ang"},
    )

    assert isfile("tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.data") as f:
        assert f.readline() == "test text\n"


def test_write_n2p2_nn(data: Data):
    """
    Test that n2p2 nn file is written successfully.
    """
    data.write_n2p2_nn(
        r_cutoff=12.0,
        type="radial",
        rule="imbalzano2018",
        mode="center",
        n_pairs=2,
        file_nn_template="input.nn.template",
        file_nn="../tests_output/input.nn",
    )

    assert isfile("tests/data/tests_output/input.nn")


def test_write_n2p2_nn_append(data: Data):
    """
    Test that n2p2 nn file is appended to successfully.
    """
    with open("tests/data/tests_output/input.nn", "w") as f:
        f.write("test text\n")
    data.write_n2p2_nn(
        r_cutoff=12.0,
        type="radial",
        rule="imbalzano2018",
        mode="center",
        n_pairs=2,
        zetas=[],
        file_nn_template="input.nn.template",
        file_nn="../tests_output/input.nn",
    )

    assert isfile("tests/data/tests_output/input.nn")
    with open("tests/data/tests_output/input.nn") as f:
        assert f.readline() == "test text\n"


def test_data_write_n2p2_scripts(data: Data):
    """
    Test that n2p2 scripts are written successfully.
    """
    data.scripts_directory = "tests/data/tests_output"
    data.write_n2p2_scripts(
        file_batch_template="../scripts/template.sh",
        file_prepare="n2p2_prune.sh",
        file_train="n2p2_train.sh",
    )

    assert isfile("tests/data/tests_output/n2p2_prune.sh")
    assert isfile("tests/data/tests_output/n2p2_train.sh")
    with open("tests/data/tests_output/n2p2_prune.sh") as f:
        assert "nnp-norm" not in f.read()


def test_data_write_n2p2_scripts_norm(data: Data):
    """
    Test that n2p2 scripts are written successfully with `normalise=True`.
    """
    data.scripts_directory = "tests/data/tests_output"
    data.write_n2p2_scripts(
        file_batch_template="../scripts/template.sh",
        file_prepare="n2p2_prune.sh",
        file_train="n2p2_train.sh",
        normalise=True,
    )

    assert isfile("tests/data/tests_output/n2p2_prune.sh")
    assert isfile("tests/data/tests_output/n2p2_train.sh")
    with open("tests/data/tests_output/n2p2_prune.sh") as f:
        assert "nnp-norm" in f.read()


def test_data_write_lammps_data(data: Data):
    """
    Test that LAMMPS data is written successfully.
    """
    data.lammps_directory = "tests/data/tests_output"
    data.write_lammps_data(
        file_xyz="cp2k_input/0.xyz",
    )

    assert isfile("tests/data/tests_output/lammps.data")


def test_min_n_config(data: Data):
    """
    Test that the correct number is returned when `n_config` and `self.trajectory` are (not)
    `None`.
    """
    assert data._min_n_config(n_provided=None) == 0
    assert data._min_n_config(n_provided=1) == 1

    data.read_trajectory(file_trajectory="trajectory.history")

    assert data._min_n_config(n_provided=None) == 2
    assert data._min_n_config(n_provided=1) == 1
    assert data._min_n_config(n_provided=3) == 2


def test_choose_weights_multiple_arguments(data: Data):
    """
    Test that an error is raised when multiple arguments provided.
    """
    with pytest.raises(ValueError) as e:
        data.choose_weights(epoch=0, minimum_criterion="RMSEpa_Etrain_pu")

    assert str(e.value) == "Both `epoch` and `minimum_criterion` provided."


def test_choose_weights_unknown_criterion(data: Data):
    """
    Test that an error is raised when an unrecognised criterion .
    """
    minimum_criterion = "unrecognisable"
    with pytest.raises(ValueError) as e:
        data.choose_weights(minimum_criterion=minimum_criterion)

    assert str(e.value) == (
        "`minimum_criterion={}` not found in `learning-curve.out` headers: "
        "['epoch', 'RMSEpa_Etrain_pu', 'RMSEpa_Etest_pu', 'RMSE_Etrain_pu', 'RMSE_Etest_pu', "
        "'MAEpa_Etrain_pu', 'MAEpa_Etest_pu', 'MAE_Etrain_pu', 'MAE_Etest_pu', "
        "'RMSE_Ftrain_pu', 'RMSE_Ftest_pu', 'MAE_Ftrain_pu', 'MAE_Ftest_pu']"
        "".format(minimum_criterion)
    )


def test_choose_weights_default(data: Data):
    """
    Test success for the default case (no arguments).
    """
    data.choose_weights()

    assert isfile("tests/data/n2p2/weights.001.data")
    try:
        with open("tests/data/n2p2/weights.001.data") as f:
            assert f.read() == "20\n"
    finally:
        remove("tests/data/n2p2/weights.001.data")


def test_choose_weights_criterion(data: Data):
    """
    Test success for the `minimum_criterion` case.
    """
    data.choose_weights(minimum_criterion="RMSEpa_Etest_pu")

    assert isfile("tests/data/n2p2/weights.001.data")
    try:
        with open("tests/data/n2p2/weights.001.data") as f:
            assert f.read() == "0\n"
    finally:
        remove("tests/data/n2p2/weights.001.data")


def test_choose_weights_epoch(data: Data):
    """
    Test success for the `epoch` case.
    """
    data.choose_weights(epoch=10)

    assert isfile("tests/data/n2p2/weights.001.data")
    try:
        with open("tests/data/n2p2/weights.001.data") as f:
            assert f.read() == "10\n"
    finally:
        remove("tests/data/n2p2/weights.001.data")


@pytest.mark.parametrize(
    "energy_threshold, force_threshold, expected_removed_indicies, stdout",
    [
        (
            np.inf,
            np.inf,
            [[]],
            "Removing outliers in tests/data/tests_output\n"
            "Removing 0 frames for having atoms outside of threshold.\n",
        ),
        (
            0.0,
            0.0,
            [[0]],
            "Removing outliers in tests/data/tests_output\n"
            "Removing 1 frames for having atoms outside of threshold.\n",
        ),
    ],
)
def test_reduce_dataset_outliers(
    data: Data,
    energy_threshold: float,
    force_threshold: float,
    expected_removed_indicies: List[int],
    stdout: str,
    capsys: pytest.CaptureFixture,
):
    """
    Test that `reduce_dataset_outliers` gives the expected outcome when removing outliers,
    and when not.
    """
    copy("tests/data/n2p2/input.data", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/output.data", "tests/data/tests_output/output.data")
    data.n2p2_directories = ["tests/data/tests_output"]
    removed_indices = data.reduce_dataset_outliers(
        energy_threshold=energy_threshold, force_threshold=force_threshold
    )

    assert removed_indices == expected_removed_indicies
    assert stdout == capsys.readouterr().out


def test_write_extrapolations_lammps_script(
    data: Data,
    capsys: pytest.CaptureFixture,
):
    """
    Test that the script is written to file successfully.
    """
    data.scripts_directory = "tests/data/tests_output"
    data.write_extrapolations_lammps_script(
        file_batch_template="../scripts/template.sh",
        file_batch_out="lammps_extrapolations.sh",
    )

    assert capsys.readouterr().out == (
        "Batch script written to tests/data/tests_output/lammps_extrapolations.sh\n"
    )
    assert isfile("tests/data/tests_output/lammps_extrapolations.sh")


def test_analyse_extrapolations(
    data: Data,
    capsys: pytest.CaptureFixture,
):
    """
    Test that the results of the extrapolations are read from file, and formatted as expected.
    """
    timestep_data = data.analyse_extrapolations()

    assert timestep_data == {
        "nve": {"mean": 484},
        "nvt": {340: 156, "mean": 156},
        "npt": {340: 255, "mean": 255},
    }
    assert capsys.readouterr().out == (
        "nve\n"
        "Temp | T_step\n"
        "MEAN |   484\n"
        "\n"
        "nvt\n"
        "Temp | T_step\n"
        " 340 |   156\n"
        "MEAN |   156\n"
        "\n"
        "npt\n"
        "Temp | T_step\n"
        " 340 |   255\n"
        "MEAN |   255\n"
        "\n"
    )


@pytest.mark.parametrize(
    "separation, expected_indices, stdout",
    [
        (0.0, [[]], "Removing 0 frames for having atoms within minimum separation.\n"),
        (
            np.inf,
            [[0]],
            "Too small interatomic distance between H-H: 4.613538961536592 Ang\n"
            "Removing 1 frames for having atoms within minimum separation.\n",
        ),
    ],
)
def test_reduce_dataset_min_separation(
    data: Data,
    separation: float,
    expected_indices: List[int],
    stdout: str,
    capsys: pytest.CaptureFixture,
):
    """
    Test that frames are removed (or not) as expected based on the separation value.
    """
    copy("tests/data/n2p2/input.data", "tests/data/tests_output/input.data")
    data.n2p2_directories = ["tests/data/tests_output"]
    structure = list(data.all_structures.values())[0]
    for species in structure.all_species:
        species.min_separation = {"H": separation, "C": separation, "O": separation}

    remove_indices = data.reduce_dataset_min_separation()

    assert remove_indices == expected_indices
    assert capsys.readouterr().out == stdout
    if len(expected_indices[0]) > 0:
        assert isfile(
            join("tests/data/tests_output/input.data.minimum_separation_backup")
        )


# QUANTUM ESPRESSO UNIT TESTS


def test_prepare_qe(data: Data):
    """
    Test that the input files and submission scripts are generated for QE in the correct
    locations.
    """
    copy("tests/data/qe/test-T300-p1.xyz", "tests/data/tests_output/test-T300-p1.xyz")
    data.scripts_directory = "tests/data/tests_output"
    data.prepare_qe(
        qe_directory="tests_output",
        temperatures=[300],
        pressures=[1],
        structure=data.all_structures["test"],
        pseudos={"H": "H.pseudo"},
    )
    assert isdir("tests/data/tests_output/T300-p1-0")
    assert isfile("tests/data/tests_output/T300-p1-0/test.in")
    assert isfile("tests/data/tests_output/T300-p1-0/pp.in")
    assert isfile("tests/data/tests_output/T300-p1-0/qe.slurm")
    assert isfile("tests/data/tests_output/qe_all.sh")


@pytest.mark.parametrize("mixing_beta", [0.25, 0.5, 0.75])
def test_write_qe_input(data: Data, mixing_beta: float):
    """
    Test that providing kwargs results in the correct value in file.
    """
    data.write_qe_input(
        atoms=Atoms("H", cell=[1, 1, 1]),
        frame_directory="tests/data/tests_output",
        structure=data.all_structures["test"],
        pseudos={"H": "H.pseudo"},
        mixing_beta=mixing_beta,
    )
    assert isfile("tests/data/tests_output/test.in")
    with open("tests/data/tests_output/test.in") as f:
        text = f.read()
        assert "mixing_beta      = {}".format(mixing_beta) in text


def test_write_qe_input_error(data: Data):
    """
    Test that an error is raised when an unrecognised kwarg is given.
    """
    with pytest.raises(ValueError) as e:
        data.write_qe_input(
            atoms=Atoms(),
            frame_directory="tests/data/tests_output",
            structure=data.all_structures["test"],
            pseudos={"H": "H.pseudo"},
            unrecognised="unrecognised",
        )

    assert str(e.value) == (
        "Key value pair {}: {} passed as **kwarg not one of the recognised options: {}"
        "".format(
            "unrecognised",
            "unrecognised",
            [
                "ibrav",
                "calculation",
                "conv_thr",
                "diago_david_ndim",
                "mixing_beta",
                "startingwfc",
                "startingpot",
                "nbnd",
                "ecutwfc",
                "ecutrho",
                "input_dft",
                "occupations",
                "degauss",
                "smearing",
                "tstress",
                "tprnfor",
                "verbosity",
                "outdir",
                "pseudo_dir",
                "disk_io",
                "restart_mode",
            ],
        )
    )


def test_write_n2p2_data_qe(data: Data):
    """
    Test that the output from QE can be read and formatted into an n2p2 "input.data" file.
    """
    copy("tests/data/qe/test-T300-p1.xyz", "tests/data/tests_output/test-T300-p1.xyz")
    mkdir("tests/data/tests_output/T300-p1-0")
    copy(
        "tests/data/qe/T300-p1-0/test.log", "tests/data/tests_output/T300-p1-0/test.log"
    )
    copy("tests/data/qe/T300-p1-0/ACF.dat", "tests/data/tests_output/T300-p1-0/ACF.dat")
    data.n2p2_directories = ["tests/data/tests_output"]
    data.all_structures["test"].get_species("H").valence = 1

    data.write_n2p2_data_qe(
        structure_name="test",
        temperatures=[300],
        pressures=[1],
        qe_directory="tests_output",
    )

    assert isfile("tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.data") as f:
        lines = f.readlines()
        assert len(lines) == 112 + 9
        for line in lines[6:-3]:
            assert line.split()[-5] != "0.0"

    data.write_n2p2_data_qe(
        structure_name="test",
        temperatures=[300],
        pressures=[1],
        qe_directory="tests_output",
    )

    assert isfile("tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.data") as f:
        lines = f.readlines()
        assert len(lines) == 2 * (112 + 9)
        for line in lines[112 + 9 + 6 : -3]:
            assert line.split()[-5] != "0.0"


def test_write_n2p2_data_qe_charge_default(data: Data):
    """
    Test that the output from QE can be read and formatted into an n2p2 "input.data" file,
    but if charges are absent then 0 is assumed.
    """
    copy("tests/data/qe/test-T300-p1.xyz", "tests/data/tests_output/test-T300-p1.xyz")
    mkdir("tests/data/tests_output/T300-p1-0")
    copy(
        "tests/data/qe/T300-p1-0/test.log", "tests/data/tests_output/T300-p1-0/test.log"
    )
    data.n2p2_directories = ["tests/data/tests_output"]

    data.all_structures["test"].get_species("H").valence = 1
    data.write_n2p2_data_qe(
        structure_name="test",
        temperatures=[300],
        pressures=[1],
        qe_directory="tests_output",
    )

    assert isfile("tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.data") as f:
        lines = f.readlines()
        assert len(lines) == 112 + 9
        for line in lines[6:-3]:
            assert float(line.split()[-5]) == 0

    data.write_n2p2_data_qe(
        structure_name="test",
        temperatures=[300],
        pressures=[1],
        qe_directory="tests_output",
    )

    assert isfile("tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.data") as f:
        lines = f.readlines()
        assert len(lines) == 2 * (112 + 9)
        for line in lines[112 + 9 + 6 : -3]:
            assert float(line.split()[-5]) == 0


@pytest.mark.parametrize(
    "remove_line, warning",
    [
        (
            "!    total energy",
            (
                "tests/data/tests_output/T300-p1-0/test.log "
                "did not complete no energy found, skipping\n"
            ),
        ),
        (
            "     Forces acting on atoms",
            (
                "tests/data/tests_output/T300-p1-0/test.log "
                "did not complete no forces found, skipping\n"
            ),
        ),
    ],
)
def test_write_n2p2_data_qe_warnings(
    data: Data, remove_line: str, warning: str, capsys: pytest.CaptureFixture
):
    """
    Test warnings are printed if log files are present, but missing energies or forces.
    """
    copy("tests/data/qe/test-T300-p1.xyz", "tests/data/tests_output/test-T300-p1.xyz")
    mkdir("tests/data/tests_output/T300-p1-0")
    copy(
        "tests/data/qe/T300-p1-0/test.log", "tests/data/tests_output/T300-p1-0/test.log"
    )
    with open("tests/data/tests_output/T300-p1-0/test.log") as f:
        lines = f.readlines()
    with open("tests/data/tests_output/T300-p1-0/test.log", "w") as f:
        for line in lines:
            if remove_line not in line:
                f.write(line)
    data.n2p2_directories = ["tests/data/tests_output"]

    with pytest.raises(OSError) as e:
        data.write_n2p2_data_qe(
            structure_name="test",
            temperatures=[300],
            pressures=[1],
            qe_directory="tests_output",
        )

    assert str(e.value) == "No files found."
    assert capsys.readouterr().out == warning


@pytest.mark.parametrize(
    "structure_name, error",
    [
        ("unrecognised", "`structure_name` unrecognised not recognized"),
        ("test", "No files found."),
    ],
)
def test_write_n2p2_data_qe_errors(data: Data, structure_name: str, error: str):
    """
    Test an error is raised if an invalid `structure_name` is provided,
    or if no filepaths can be found.
    """
    copy("tests/data/qe/test-T300-p1.xyz", "tests/data/tests_output/test-T300-p1.xyz")
    with pytest.raises(Exception) as e:
        data.write_n2p2_data_qe(
            structure_name=structure_name,
            temperatures=[300],
            pressures=[1],
            qe_directory="tests_output",
        )

    assert str(e.value) == error


def test_remove_n2p2_normalisation(data: Data):
    """
    Test that the files are moved and removed correctly.
    """
    with open("tests/data/tests_output/input.nn", "w") as f:
        f.write("Normalised file.")
    with open("tests/data/tests_output/input.nn.bak", "w") as f:
        f.write("Unnormalised file.")
    with open("tests/data/tests_output/output.data", "w") as f:
        f.write("Output file.")
    with open("tests/data/tests_output/evsv.dat", "w") as f:
        f.write("EVSV file.")
    data.n2p2_directories = ["tests/data/tests_output"]

    data.remove_n2p2_normalisation()

    assert not isfile("tests/data/tests_output/input.nn.bak")
    assert not isfile("tests/data/tests_output/output.data")
    assert not isfile("tests/data/tests_output/evsv.dat")
    assert isfile("tests/data/tests_output/input.nn")
    with open("tests/data/tests_output/input.nn") as f:
        assert f.read() == "Unnormalised file."
