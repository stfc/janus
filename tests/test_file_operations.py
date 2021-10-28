"""
Unit tests for `file_operations.py`
"""
import numpy as np
import pytest

from cc_hdnnp.file_operations import (
    format_template_file,
    read_lammps_log,
    read_last_timestep,
)


@pytest.mark.parametrize(
    "format_shell_variables, text_in, text_out",
    [
        (True, "{normal} ${shell}", "one two"),
        (False, "{normal} ${shell}", "one $two"),
    ],
)
def test_format_template_file(
    format_shell_variables: bool, text_in: str, text_out: str
):
    """
    Test that the file is formatted correctly depending on `format_shell_variables`.
    """
    template_file = "tests/data/tests_output/template.txt"
    formatted_file = "tests/data/tests_output/formatted.txt"
    with open(template_file, "w") as f:
        f.write(text_in)

    format_template_file(
        template_file=template_file,
        formatted_file=formatted_file,
        format_dict={"normal": "one", "shell": "two"},
        format_shell_variables=format_shell_variables,
    )

    with open(formatted_file) as f:
        text = f.read()
    assert text == text_out


@pytest.mark.parametrize("extrapolation_free_timesteps_expected", [162, 100])
def test_read_lammps_log(extrapolation_free_timesteps_expected: int):
    """Test that for a mocked lammps.log file, we read the expected number of timesteps."""
    if extrapolation_free_timesteps_expected > 161:
        extrapolation_free_lines_expected = -1
    else:
        extrapolation_free_lines_expected = 3613 + extrapolation_free_timesteps_expected
    timesteps_counter = -3
    text = ""
    log_lammps_file = "tests/data/tests_output/log.lammps"
    with open(
        "tests/data/active_learning/mode1/test_npt_hdnnp1_t325_p1.0_1/log.lammps"
    ) as f:
        for line in f.readlines():
            if (
                timesteps_counter > extrapolation_free_timesteps_expected
                or not line.startswith("### NNP EXTRAPOLATION WARNING ###")
            ):
                text += line
            if line.startswith("thermo"):
                timesteps_counter += 1

    with open(log_lammps_file, "w") as f:
        f.write(text)

    (
        timesteps,
        extrapolation_free_lines,
        extrapolation_free_timesteps,
    ) = read_lammps_log(dump_lammpstrj=1, log_lammps_file=log_lammps_file)

    assert all(timesteps == np.arange(1, 162))
    assert extrapolation_free_lines == extrapolation_free_lines_expected
    assert extrapolation_free_timesteps == min(
        extrapolation_free_timesteps_expected, 161
    )


def test_read_lammps_log_errors():
    """Test that we raise a ValueError for an empty log.lammps file."""
    log_lammps_file = "tests/data/tests_output/log.lammps"
    with open(log_lammps_file, "w"):
        pass

    with pytest.raises(ValueError) as e:
        read_lammps_log(dump_lammpstrj=1, log_lammps_file=log_lammps_file)

    assert str(e.value) == "{} was empty".format(log_lammps_file)


def test_read_last_timestep():
    """Test that we raise a ValueError if a timestep cannot be found."""
    log_lammps_file = "tests/data/tests_output/log.lammps"
    with open(log_lammps_file, "w") as f:
        f.write("\ntext\n")

    with pytest.raises(ValueError) as e:
        read_last_timestep(file_lammps=log_lammps_file)

    assert str(e.value) == "Could not extract final timestep from {}".format(
        log_lammps_file
    )
