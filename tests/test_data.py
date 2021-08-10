"""
Unit tests for `data.py`
"""

from os import listdir, remove
from os.path import isfile
from shutil import copy

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
        structures=AllStructures(structure), main_directory="tests/data", n2p2_bin="",
        lammps_executable=""
    )

    for file in listdir("tests/data/tests_output"):
        remove("tests/data/tests_output/" + file)


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


def test_data_convert_active_learning_to_xyz(data: Data):
    """
    Test that active learning structures can be written in xyz format.
    """
    data.convert_active_learning_to_xyz(
        file_structure="input.data-add", file_xyz="tests_output/{}.xyz"
    )

    assert isfile("tests/data/tests_output/0.xyz")


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


def test_data_write_cp2k(data: Data):
    """
    Test that cp2k input and batch scripts are written to file.
    """
    # Copy template to the output directory to prevent removing them during clearup.
    copy("tests/data/cp2k_input/template.inp", "tests/data/tests_output/template.inp")
    copy("tests/data/cp2k_input/template.sh", "tests/data/tests_output/template.sh")

    data.write_cp2k(
        file_bash="tests_output/all.sh",
        file_batch="tests_output/{}.sh",
        file_input="tests_output/{}.inp",
        file_xyz="cp2k_input/{}.xyz",
        n_config=1,
    )

    assert isfile("tests/data/tests_output/all.sh")
    assert isfile("tests/data/tests_output/n_0.sh")
    assert isfile("tests/data/tests_output/n_0.inp")


def test_data_write_cp2k_kwargs(data: Data):
    """
    Test that cp2k input and batch scripts are written to file with n_config, cutoff and
    relcutoff given.
    """
    # Copy template to the output directory to prevent removing them during clearup.
    copy("tests/data/cp2k_input/template.inp", "tests/data/tests_output/template.inp")
    copy("tests/data/cp2k_input/template.sh", "tests/data/tests_output/template.sh")

    data.write_cp2k(
        file_bash="tests_output/all.sh",
        file_batch="tests_output/{}.sh",
        file_input="tests_output/{}.inp",
        file_xyz="cp2k_input/{}.xyz",
        n_config=1,
        cutoff=(60.0,),
        relcutoff=(600.0,),
    )

    assert isfile("tests/data/tests_output/all.sh")
    assert isfile("tests/data/tests_output/n_0_cutoff_60.0_relcutoff_600.0.sh")
    assert isfile("tests/data/tests_output/n_0_cutoff_60.0_relcutoff_600.0.inp")


def test_data_write_cp2k_kwargs_floats(data: Data):
    """
    Test that cp2k input and batch scripts are written to file with n_config, cutoff and
    relcutoff given as floats.
    """
    # Copy template to the output directory to prevent removing them during clearup.
    copy("tests/data/cp2k_input/template.inp", "tests/data/tests_output/template.inp")
    copy("tests/data/cp2k_input/template.sh", "tests/data/tests_output/template.sh")

    data.write_cp2k(
        file_bash="tests_output/all.sh",
        file_batch="tests_output/{}.sh",
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
    data.write_n2p2_data(
        structure_name="test",
        file_cp2k_out="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
        file_xyz="cp2k_input/{}.xyz",
        file_n2p2_input="tests_output/input.data",
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
    data.write_n2p2_data(
        structure_name="test",
        file_cp2k_out="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
        file_xyz="cp2k_input/{}.xyz",
        file_n2p2_input="tests_output/input.data",
        n_config=1,
        n2p2_units={"length": "Ang", "energy": "eV", "force": "eV / Ang"},
    )

    assert isfile("tests/data/tests_output/input.data")


def test_data_write_n2p2_data_appended(data: Data):
    """
    Test that n2p2 data is appended to file without overwrite if output file already exists.
    """
    with open("tests/data/tests_output/input.data", "w") as f:
        f.write("test text\n")

    data.write_n2p2_data(
        structure_name="test",
        file_cp2k_out="cp2k_output/n_0_cutoff_600_relcutoff_60.log",
        file_cp2k_forces="cp2k_output/n_0_cutoff_600_relcutoff_60-forces-1_0.xyz",
        file_xyz="cp2k_input/{}.xyz",
        file_n2p2_input="tests_output/input.data",
        n_config=1,
        n2p2_units={"length": "Ang", "energy": "eV", "force": "eV / Ang"},
    )

    assert isfile("tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.data") as f:
        assert f.readline() == "test text\n"


def test_data_write_n2p2_nn(data: Data):
    """
    Test that n2p2 nn file is written successfully.
    """
    data.write_n2p2_nn(
        r_cutoff=12.0,
        type="radial",
        rule="imbalzano2018",
        mode="center",
        n_pairs=2,
        file_nn_template="n2p2/input.nn.template",
        file_nn="tests_output/input.nn",
    )

    assert isfile("tests/data/tests_output/input.nn")


def test_data_write_n2p2_nn_append(data: Data):
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
        file_nn_template="n2p2/input.nn.template",
        file_nn="tests_output/input.nn",
    )

    assert isfile("tests/data/tests_output/input.nn")
    with open("tests/data/tests_output/input.nn") as f:
        assert f.readline() == "test text\n"


def test_data_write_n2p2_scripts(data: Data):
    """
    Test that n2p2 scripts are written successfully.
    """
    data.write_n2p2_scripts(
        file_batch_template="n2p2/template.sh",
        file_prune="tests_output/n2p2_prune.sh",
        file_train="tests_output/n2p2_train.sh",
    )

    assert isfile("tests/data/tests_output/n2p2_prune.sh")
    assert isfile("tests/data/tests_output/n2p2_train.sh")


def test_data_write_lammps_data(data: Data):
    """
    Test that LAMMPS data is written successfully.
    """
    data.write_lammps_data(
        file_xyz="cp2k_input/0.xyz",
        file_data="tests_output/lammps.data",
    )

    assert isfile("tests/data/tests_output/lammps.data")


def test_data_write_lammps_pair(data: Data):
    """
    Test that LAMMPS data is written successfully.
    """
    data.write_lammps_pair(
        r_cutoff=6.35,
        file_lammps_template="lammps/template.lmp",
        file_out="tests_output/md.lmp",
    )

    assert isfile("tests/data/tests_output/md.lmp")


def test_data_write_lammps_pair_units(data: Data):
    """
    Test that LAMMPS data is written successfully with custom units.
    # TODO generalise to all possible LAMMPS units
    """
    data.write_lammps_pair(
        r_cutoff=6.35,
        file_lammps_template="lammps/template.lmp",
        file_out="tests_output/md.lmp",
        lammps_unit_style="metal",
    )

    assert isfile("tests/data/tests_output/md.lmp")
    with open("tests/data/tests_output/md.lmp") as f:
        lines = f.readlines()
        assert "cflength 1.8897261258369282 cfenergy 0.03674932247495664" in lines[20]


def test_data_write_lammps_pair_unknown_units(data: Data):
    """
    Test that an error is raised when given unrecognised units.
    """
    lammps_unit_style = "bad_units"
    with pytest.raises(ValueError) as e:
        data.write_lammps_pair(
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
