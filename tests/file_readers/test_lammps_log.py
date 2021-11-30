"""
Unit tests for `file_readers/lammps_log.py`
"""
import numpy as np
import pytest

from cc_hdnnp.file_readers import (
    read_lammps_log,
)


def test_read_lammps_log_complete():
    """Test that for a compeleted lammps.log file, we read the expected number of timesteps."""
    log_lammps_file = "tests/data/lammps/nve_complete.log"
    (
        timesteps,
        extrapolation_free_lines,
        extrapolation_free_timesteps,
        temperatures,
    ) = read_lammps_log(dump_lammpstrj=1, log_lammps_file=log_lammps_file)

    assert all(timesteps == np.arange(50001))
    assert extrapolation_free_lines == -1
    assert extrapolation_free_timesteps == 50000
    assert len(temperatures) == 50001


@pytest.mark.parametrize("extrapolation_free_timesteps_expected", [162, 100])
def test_read_lammps_log(extrapolation_free_timesteps_expected: int):
    """Test that for a mocked lammps.log file, we read the expected number of timesteps."""
    if extrapolation_free_timesteps_expected > 161:
        extrapolation_free_lines_expected = -1
    else:
        extrapolation_free_lines_expected = 3614 + extrapolation_free_timesteps_expected
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
        temperatures,
    ) = read_lammps_log(dump_lammpstrj=1, log_lammps_file=log_lammps_file)

    assert len(timesteps) == 162
    assert all(timesteps == np.arange(0, 162))
    assert extrapolation_free_lines == extrapolation_free_lines_expected
    assert extrapolation_free_timesteps == min(
        extrapolation_free_timesteps_expected, 161
    )
    assert len(temperatures) == 162


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
        read_lammps_log(log_lammps_file=log_lammps_file, dump_lammpstrj=1)

    assert str(e.value) == (
        f"{log_lammps_file} does not contain simulation headers, indicating it crashed "
        "before the 0th timestep."
    )
