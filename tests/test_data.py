"""
Unit tests for `data.py`
"""

from os import listdir, mkdir, remove
from os.path import isfile, join
from shutil import copy, rmtree
from typing import Dict, List, Literal, Union

from genericpath import isdir
import numpy as np
import pytest

from cc_hdnnp.data import Data
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
    if needed, and a trailing newline if writing multiple frames to one file.
    """
    data.convert_active_learning_to_xyz(
        file_structure="input.data-add",
        file_xyz=file_xyz,
        single_output=single_output,
    )

    file_out = "tests/data/" + file_xyz.format(0)
    assert isfile(file_out)
    with open(file_out) as f:
        text = f.read()
    assert text.endswith("\n") == single_output


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


def test_data_format_lammps_input(data: Data):
    """
    Test that LAMMPS data is written successfully.
    """
    data.format_lammps_input(
        structure=data.all_structures["test"],
        n_steps=1000,
        r_cutoff=6.35,
        file_lammps_template="lammps/template.lmp",
        file_out="tests_output/md.lmp",
    )

    assert isfile("tests/data/tests_output/md.lmp")
    with open("tests/data/tests_output/md.lmp") as f:
        lines = f.readlines()
    assert "units electron\n" == lines[7]
    assert "mass 1 1.0\n" == lines[11]
    assert (
        'pair_style nnp dir n2p2 showew no showewsum 10 resetew no maxew 100 emap "1:H"\n'
        == lines[19]
    )
    assert "pair_coeff * * 6.35\n" == lines[20]
    assert "dump_modify     cust element  H\n" == lines[27]
    assert "run 1000\n" == lines[32]


@pytest.mark.parametrize(
    "lammps_unit_style, cflength, cfenergy",
    [
        ("real", 1.8897261258369282, 0.001593601438080425),
        ("metal", 1.8897261258369282, 0.03674932247495664),
        ("si", 18897261258.369282, 2.2937123159746854e17),
        ("cgs", 188972612.58369282, 22937123159.746857),
        ("micro", 18897.261258369283, 229371.23159746855),
        ("nano", 18.897261258369284, 0.22937123159746856),
    ],
)
def test_data_format_lammps_input_units(
    data: Data, lammps_unit_style: str, cflength: float, cfenergy: float
):
    """
    Test that LAMMPS data is written successfully with custom units
    (i.e. not the default "electron").
    """
    data.format_lammps_input(
        structure=data.all_structures["test"],
        n_steps=1000,
        r_cutoff=6.35,
        file_lammps_template="lammps/template.lmp",
        file_out="tests_output/md.lmp",
        lammps_unit_style=lammps_unit_style,
    )

    assert isfile("tests/data/tests_output/md.lmp")
    with open("tests/data/tests_output/md.lmp") as f:
        lines = f.readlines()
    assert "units {}\n".format(lammps_unit_style) == lines[7]
    assert "cflength {0} cfenergy {1}".format(cflength, cfenergy) in lines[19]


def test_data_format_lammps_input_unknown_units(data: Data):
    """
    Test that an error is raised when given unrecognised units.
    """
    lammps_unit_style = "bad_units"
    with pytest.raises(ValueError) as e:
        data.format_lammps_input(
            structure=data.all_structures["test"],
            n_steps=1000,
            r_cutoff=6.35,
            file_lammps_template="lammps/template.lmp",
            file_out="tests_output/md.lmp",
            lammps_unit_style=lammps_unit_style,
        )

    assert str(e.value) == "`lammps_unit_style={}` not recognised".format(
        lammps_unit_style
    )


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
    "energy_threshold, force_threshold, stdout",
    [(np.inf, np.inf, "0 outliers found: []\n"), (0.0, 0.0, "1 outliers found: [0]\n")],
)
def test_remove_outliers(
    data: Data,
    energy_threshold: float,
    force_threshold: float,
    stdout: str,
    capsys: pytest.CaptureFixture,
):
    """
    Test that `remove_outliers` gives the exptected outcome when removing outliers, and when
    not.
    """
    copy("tests/data/n2p2/input.data", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/output.data", "tests/data/tests_output/output.data")
    data.n2p2_directories = ["tests/data/tests_output"]
    data.remove_outliers(
        energy_threshold=energy_threshold, force_threshold=force_threshold
    )

    assert stdout in capsys.readouterr().out


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
    assert timestep_data == {
        "nve": {"mean": 484},
        "nvt": {340: 156, "mean": 156},
        "npt": {340: 255, "mean": 255},
    }


@pytest.mark.parametrize(
    "separation, expected_indices, stdout",
    [
        (
            0.0,
            [[]],
            "Removing 0 frames for having atoms within minimum separation.\n"
            "No frames to remove\n",
        ),
        (
            np.inf,
            [[0]],
            "Removing 1 frames for having atoms within minimum separation.\n",
        ),
    ],
)
def test_trim_dataset_separation(
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

    remove_indices = data.trim_dataset_separation(
        structure,
    )

    assert remove_indices == expected_indices
    assert capsys.readouterr().out == stdout
    if len(expected_indices[0]) > 0:
        assert isfile(
            join("tests/data/tests_output/input.data.minimum_separation_backup")
        )


@pytest.mark.parametrize(
    "criteria, differences",
    [
        (
            0.5,
            {
                "0[2]": 1.9442222,
                "1[2]": 0.6480741,
                "1[0]": 1.2961481,
                "1[0 2]": 0.6480741,
                "2[0]": 1.9442222,
            },
        ),
        (
            "mean",
            {
                "0[2]": 1.9502137,
                "1[2]": 0.6658329,
                "1[0]": 1.3051181,
                "1[0 2]": 0.6658329,
                "2[0]": 1.9502137,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "starting_frame_indices, settings, scaling, expected_starting_indices, "
    "expected_proposed_indices, expected_selected_indices, select_extreme_frames",
    [
        (
            None,
            "",
            "1 1 -1.0 -1.0 0.5 1.0\n1 2 -1.0 -1.0 1.0 2.0\n1 3 -1.0 -1.0 1.5 3.0\n",
            np.array([2]),
            np.array([1, 0]),
            np.array([2, 0]),
            False,
        ),
        (
            [2],
            "",
            "1 1 -1.0 -1.0 0.5 1.0\n1 2 -1.0 -1.0 1.0 2.0\n1 3 -1.0 -1.0 1.5 3.0\n",
            np.array([2]),
            np.array([1, 0]),
            np.array([2, 0]),
            False,
        ),
        (
            None,
            "",
            "1 1 0.0 1.0 0.5 1.0\n1 2 0.0 2.0 1.0 2.0\n1 3 0.0 3.0 1.5 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
        (
            [2],
            "",
            "1 1 0.0 1.0 0.5 1.0\n1 2 0.0 2.0 1.0 2.0\n1 3 0.0 3.0 1.5 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
        (
            None,
            "scale_symmetry_functions_sigma 1",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0]),
            np.array([2, 1]),
            np.array([0, 2]),
            True,
        ),
        (
            [2],
            "scale_symmetry_functions_sigma 1",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
        (
            None,
            "scale_symmetry_functions 1\nscale_min_short -0.3\nscale_max_short 0.3",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0]),
            np.array([2, 1]),
            np.array([0, 2]),
            True,
        ),
        (
            [2],
            "scale_symmetry_functions 1\nscale_min_short -0.3\nscale_max_short 0.3",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
    ],
)
def test_rebuild_dataset(
    data: Data,
    capsys: pytest.CaptureFixture,
    settings: str,
    scaling: str,
    expected_starting_indices: List[int],
    expected_proposed_indices: List[int],
    expected_selected_indices: List[int],
    criteria: Union[float, Literal["mean"]],
    differences: Dict[str, float],
    starting_frame_indices: List[int],
    select_extreme_frames: bool,
):
    """
    Test that the expected frame is removed, with the expected metric value(s),
    for a given input.
    """
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    copy("tests/data/n2p2/input.data", "tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.nn", "w") as f:
        f.write(settings)
    with open("tests/data/tests_output/scaling.data", "w") as f:
        f.write(scaling)
    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]
    difference = np.array(
        [
            differences["{0}{1}".format(i, expected_starting_indices)]
            for i in expected_proposed_indices
        ]
    )

    selected_indices = data.rebuild_dataset(
        atoms_per_frame=2,
        n_frames_to_select=1,
        n_frames_to_compare=1,
        n_frames_to_propose=2,
        seed=0,
        criteria=criteria,
        starting_frame_indices=starting_frame_indices,
        select_extreme_frames=select_extreme_frames,
    )

    text = capsys.readouterr().out
    assert "Values read from file in " in text
    assert (
        "Starting rebuild with the following frames selected:\n{}\n".format(
            expected_starting_indices
        )
        in text
    )
    assert "Proposed indices:\n{}\n".format(expected_proposed_indices) in text
    assert (
        "Difference metric summed over all elements:\n{}\n".format(difference) in text
    )
    assert (
        "Selected indices:\n{}\n".format(
            [expected_proposed_indices[np.argmax(difference)]]
        )
        in text
    )
    assert "Time taken: " in text
    assert np.all(selected_indices == expected_selected_indices)


@pytest.mark.parametrize(
    "criteria, settings, error",
    [
        (
            1.5,
            "",
            "`criteria` must be between 0 and 1, but was 1.5",
        ),
        (
            "mode",
            "",
            "`criteria` must be a quantile (float) between 0 and 1 or 'mean', but was mode",
        ),
        (
            0.5,
            "scale_symmetry_functions True\nscale_symmetry_functions_sigma True\n",
            "Both scale_symmetry_functions and scale_symmetry_functions_sigma "
            "were present in settings file.",
        ),
        (
            0.5,
            "scale_symmetry_functions True\n",
            "If scale_symmetry_functions is set, both scale_min_short and "
            "scale_max_short must be present.",
        ),
    ],
)
def test_rebuild_dataset_errors(
    data: Data,
    criteria: Union[float, Literal["mean"]],
    settings: str,
    error: str,
):
    """
    Test that the expected errors are raised by giving the incorrect `critera` or `settings`.
    """
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    copy("tests/data/n2p2/input.data", "tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.nn", "w") as f:
        f.write(settings)
    with open("tests/data/tests_output/scaling.data", "w") as f:
        f.write("1 1 0.0 1.0 0.5 1.0\n1 2 0.0 2.0 1.0 2.0\n1 3 0.0 3.0 1.5 3.0\n")
    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    with pytest.raises(ValueError) as e:
        data.rebuild_dataset(
            atoms_per_frame=2,
            n_frames_to_select=1,
            n_frames_to_compare=1,
            n_frames_to_propose=2,
            seed=0,
            criteria=criteria,
        )

    assert str(e.value) == error


# CLUSTERING UNIT TESTS


@pytest.mark.parametrize(
    "element, compare_atomic, labels",
    [("H", True, " -1 -1\n -1 -1\n -1 -1\n"), ("all", False, " -1\n -1\n -1\n")],
)
def test_cluster_dataset(
    data: Data,
    capsys: pytest.CaptureFixture,
    element: str,
    compare_atomic: bool,
    labels: str,
):
    """
    Test that clustering results in the correct information being printed and written to file,
    whatever the value of `compare_atomic`.
    """
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    atom_environments = data.cluster_dataset(
        atoms_per_frame=2,
        compare_atomic=compare_atomic,
        file_out="tests_output/clustered_{}.data",
    )

    assert list(atom_environments.keys()) == ["H"]
    assert np.all(
        atom_environments["H"]
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
    if compare_atomic:
        assert "Element: {}\n".format(element) in text
    assert "0 labels assigned\n" in text
    assert "Noise    :      " in text
    assert "Clustered in " in text


# CUR UNIT TESTS

@pytest.mark.parametrize(
    "file_dict",
    [{}, {"file_in": "input.nn", "file_out": "input.nn", "file_backup": "input.nn.CUR_backup"}],
)
def test_decompose_dataset_symf(
    data: Data,
    capsys: pytest.CaptureFixture,
    file_dict: Dict[str, str],
):
    """
    Test that running CUR decomposition in "symf" mode results in the expected information
    being printed and written to file.
    """
    copy("tests/data/n2p2/input.nn", "tests/data/tests_output/input.nn")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    with open("tests/data/tests_output/input.nn", "a") as f:
        f.write("symfunction_short H 0\nsymfunction_short H 1\nsymfunction_short H 2\n")

    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    atom_environments = data.decompose_dataset(
        atoms_per_frame=2,
        selection_mode="symf",
        **file_dict,
    )

    assert list(atom_environments.keys()) == ["H"]
    assert np.all(
        atom_environments["H"]
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
    assert "Element: H\n" in text
    assert "Selecting 1 out of 3 symmetry functions\n" in text
    assert "Selected indices in " in text
    assert "[2]\n" in text

    file = "tests/data/tests_output/input.nn"
    assert isfile(file + ".CUR_backup")
    with open(file + ".CUR_backup") as f:
        lines = f.readlines()
        assert lines[-3] == "symfunction_short H 0\n"
        assert lines[-2] == "symfunction_short H 1\n"
        assert lines[-1] == "symfunction_short H 2\n"

    assert isfile(file)
    with open(file) as f:
        lines = f.readlines()
        assert lines[-3] == "#symfunction_short H 0\n"
        assert lines[-2] == "#symfunction_short H 1\n"
        assert lines[-1] == "symfunction_short H 2\n"


@pytest.mark.parametrize(
    "file_dict",
    [{}, {"file_in": "input.data", "file_out": "input.data", "file_backup": "input.data.CUR_backup"}],
)
def test_decompose_dataset_data(
    data: Data,
    capsys: pytest.CaptureFixture,
    file_dict: Dict[str, str],
):
    """
    Test that running CUR decomposition in "data" mode results in the expected information
    being printed and written to file.
    """
    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")

    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    atom_environments = data.decompose_dataset(
        atoms_per_frame=2,
        selection_mode="data",
        **file_dict,
    )

    assert list(atom_environments.keys()) == ["H"]
    assert np.all(
        atom_environments["H"]
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
        assert len(lines) == 27

    assert isfile(file)
    with open(file) as f:
        lines = f.readlines()
        assert len(lines) == 9
        assert lines[-2] == "charge  2.0\n"


def test_decompose_dataset_error(
    data: Data,
):
    """
    Test that running CUR decomposition with an unrecognised argument for `selection_mode`
    raises a ValueError.
    """
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    with pytest.raises(ValueError) as e:
        data.decompose_dataset(
            atoms_per_frame=2,
            selection_mode="unrecognised",
        )

    assert str(e.value) == "`selection_mode` must be one of 'symf', 'data' but was unrecognised"

@pytest.mark.parametrize(
    "n_to_select, error",
    [
        (
            101,
            "`n_to_select` must be less than the number of quantity, but they were 101, 100"
        ),
        (
            0,
            "`n_to_select` must be at least 1, but was 0"
        )
    ],
)
def test_validate_n_to_select(
    data: Data,
    n_to_select: int,
    error: str,
):
    """
    Test ValueErrors are raised if `n_to_select` is greater than `n_total` or less than 0.
    """
    with pytest.raises(ValueError) as e:
        data._validate_n_to_select(
            n_to_select=n_to_select,
            n_total=100,
            quantitiy="quantity",
        )

    assert str(e.value) == error


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
        pseudos=["H.pseudo"],
    )
    assert isdir("tests/data/tests_output/T300-p1-0")
    assert isfile("tests/data/tests_output/T300-p1-0/test.in")
    assert isfile("tests/data/tests_output/T300-p1-0/pp.in")
    assert isfile("tests/data/tests_output/T300-p1-0/qe.slurm")
    assert isfile("tests/data/tests_output/qe_all.sh")


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
    data.all_structures["test"].all_species.get_species("H").valence = 1

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

    data.all_structures["test"].all_species.get_species("H").valence = 1
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
            assert line.split()[-5] == "0.0"

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
            assert line.split()[-5] == "0.0"


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

    with pytest.raises(Exception) as e:
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
