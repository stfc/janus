"""
Unit tests for `lammps_input.py`
"""

from os import remove
from os.path import isfile

import pytest

from janus.lammps_input import format_lammps_input


@pytest.mark.parametrize("integrator", ["nve", "nvt", "npt"])
def test_format_lammps_input(integrator: str):
    """
    Test single arguments for `n_steps`, `integrators` and `temps`
    are gracefully handled.
    """
    try:
        format_lammps_input(
            formatted_file="tests/data/tests_output/lammps.md",
            masses="mass 1 1.00794\n",
            emap="1:H",
            template_file="tests/data/lammps/lammps_template.md",
            n_steps=1,
            integrators=integrator,
            temps="300",
            elements="H",
        )
        assert isfile("tests/data/tests_output/lammps.md")
    finally:
        remove("tests/data/tests_output/lammps.md")


def test_format_lammps_input_integrator_error():
    """
    Test a ValueError is raised for an unrecognised `integrators` argument.
    """
    with pytest.raises(ValueError) as e:
        format_lammps_input(
            formatted_file="tests/data/tests_output/lammps.md",
            masses="mass 1 1.00794\n",
            emap="1:H",
            n_steps=1,
            integrators="unrecognised",
            temps="300",
            elements="H",
        )

    assert (
        str(e.value)
        == "`integrator` must be one of 'nve', 'nvt' or 'npt', but was 'unrecognised'"
    )


def test_format_lammps_input_barostat_error():
    """
    Test a ValueError is raised for an unrecognised `barostat` argument.
    """
    with pytest.raises(ValueError) as e:
        format_lammps_input(
            formatted_file="tests/data/tests_output/lammps.md",
            masses="mass 1 1.00794\n",
            emap="1:H",
            n_steps=1,
            integrators="npt",
            temps="300",
            barostat="unrecognised",
            elements="H",
        )

    assert str(e.value) == "Barostat option 'unrecognised' is not implemented."


def test_format_lammps_input_elements_error():
    """
    Test a ValueError if neither `elements` nor `dump_commands` provided.
    """
    with pytest.raises(ValueError) as e:
        format_lammps_input(
            formatted_file="tests/data/tests_output/lammps.md",
            masses="mass 1 1.00794\n",
            emap="1:H",
            n_steps=1,
            integrators="nve",
            temps="300",
        )

    assert (
        str(e.value)
        == "Either `dump_commands` or `elements` must not be None, but both were."
    )
