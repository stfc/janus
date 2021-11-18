"""
Unit tests for `structure.py`
"""

from os.path import isfile
from typing import List, Tuple

import numpy as np
import pytest

from cc_hdnnp.dataset import Dataset, Frame
from cc_hdnnp.structure import AllStructures, Species, Structure


def test_frame_units_error():
    """Test that attempting to set units with unsupported keys raises an ValueError"""
    with pytest.raises(ValueError) as e:
        Frame(
            lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            units={"test": "test"},
        )

    assert str(e.value) == (
        "Only 'energy', 'length' are supported keys in `units`, "
        "but {'test': 'test'} was provided."
    )


@pytest.mark.parametrize(
    "energy_threshold, energy_accepted",
    [((0, 0), False), ((-10, 10), True)],
)
@pytest.mark.parametrize(
    "force_threshold, force_accepted",
    [(0, False), (10, True)],
)
def test_frame_check_threshold(
    energy_threshold: Tuple[float, float],
    energy_accepted: bool,
    force_threshold: float,
    force_accepted: bool,
):
    """Test that the correct result is returned for different thresholds"""
    frame = Frame(
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbols=["H"],
        positions=np.array([[0.0, 0.0, 0.0]]),
        energy=1,
        forces=np.array([[1.0, 0.0, 0.0]]),
    )

    assert frame.check_threshold(
        energy_threshold=energy_threshold, force_threshold=force_threshold
    ) == (energy_accepted and force_accepted)


def test_dataset():
    """
    Test that we can init a Dataset from file and check its properties.
    """
    dataset = Dataset(data_file="tests/data/n2p2/input.data")
    assert len(dataset) == 1
    assert dataset.all_lattices.shape == (1, 3, 3)
    assert dataset.all_positions.shape == (1, 512, 3)
    assert dataset.all_symbols.shape == (1, 512)
    assert dataset.all_forces.shape == (1, 512, 3)
    assert dataset.all_charges.shape == (1, 512)
    assert dataset.all_energies != 0
    assert dataset.all_names == np.array(["test"])
    assert dataset.all_statistics == ["stats"]
    assert dataset[0].frame_file == "test_s0"
    assert dataset[0].units == {"energy": "Ha", "length": "Bohr"}


def test_dataset_lammps():
    """
    Test that we can init a Dataset from a non "n2p2" format and check its properties.
    """
    dataset = Dataset(
        data_file="tests/data/lammps/lammps.data", format="lammps-data", style="atomic"
    )
    assert len(dataset) == 1
    assert dataset.all_lattices.shape == (1, 3, 3)
    assert dataset.all_positions.shape == (1, 512, 3)
    assert dataset.all_symbols.shape == (1, 512)
    assert dataset.all_forces.shape == (1, 512, 3)
    assert dataset.all_charges.shape == (1, 512)
    assert dataset.all_energies == 0
    assert dataset.all_names == np.array([None])
    assert dataset.all_statistics == np.array([None])
    assert dataset[0].frame_file == np.array([None])
    assert dataset[0].units == {"energy": "eV", "length": "Ang"}


def test_dataset_error():
    """
    Test that a TypeError is raised if a list of anything other than Frame objects
    is provided.
    """
    with pytest.raises(TypeError) as e:
        Dataset(frames=["text"])

    assert (
        str(e.value)
        == "`frames` must be a list of Frame objects, but had entries with type <class 'str'>"
    )


def test_dataset_units():
    """
    Test that we can init a Dataset from file and check its units.
    """
    dataset = Dataset(data_file="tests/data/n2p2/input.data.metal_units")
    assert len(dataset) == 1
    assert dataset[0].units == {"energy": "eV", "length": "Ang"}


def test_dataset_naming():
    """
    Test that a loaded Dataset has the names set for Frames based on file comments.
    """
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    structure_0 = Structure(
        name="test0", all_species=[species_1], delta_E=None, delta_F=None
    )
    structure_1 = Structure(
        name="test1", all_species=[species_1], delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test2", all_species=[species_1], delta_E=None, delta_F=None
    )
    all_structures = AllStructures(structure_0, structure_1, structure_2)

    dataset = Dataset(
        data_file="tests/data/n2p2/input.data.CUR", all_structures=all_structures
    )

    assert len(dataset) == 3
    for i, frame in enumerate(dataset):
        assert frame.name == "test{}".format(i)


def test_dataset_check_min_separation_all_error():
    """
    Test that a ValueError is raised when calling `check_min_separation`
    without `all_structures` being set.
    """
    dataset = Dataset(data_file="tests/data/n2p2/input.data")

    with pytest.raises(ValueError) as e:
        next(dataset.check_min_separation_all())

    assert str(e.value) == "Cannot check separation if `all_structures` is not set."


@pytest.mark.parametrize(
    "symbols",
    [
        ["H", "He"],
        ["Li", "Be"],
    ],
)
def test_check_nearest_neighbours(
    symbols: List[str],
):
    """
    Test that we accept the neighbours in the cases
    where they are empty or only have 1 dimension.
    """
    frame = Frame(
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbols=symbols,
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    )
    accepted, d = frame.check_nearest_neighbours(
        element_i="H",
        element_j="He",
        d_min=0.1,
    )

    assert accepted
    assert d == -1


@pytest.mark.parametrize(
    "format, lines",
    [
        ("n2p2", 9),
        ("lammps-data", 12),
    ],
)
def test_write(
    format: str,
    lines: int,
):
    """
    Test we write correctly in multiple formats without conditions.
    """
    file_path = "tests/data/tests_output/test.txt"
    frame = Frame(
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbols=["H"],
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    dataset = Dataset(frames=[frame])
    dataset.write(file_out=file_path, format=format)

    assert isfile(file_path)
    with open(file_path) as f:
        assert len(f.readlines()) == lines


@pytest.mark.parametrize(
    "format, lines",
    [
        ("n2p2", 9),
        ("lammps-data", 12),
    ],
)
def test_write_conditions(
    format: str,
    lines: int,
):
    """
    Test we write correctly in multiple formats with conditions.
    """
    file_path_1 = "tests/data/tests_output/1.txt"
    file_path_2 = "tests/data/tests_output/2.txt"
    conditions_1 = [True, False]
    conditions_2 = (b for b in conditions_1)
    frame_1 = Frame(
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbols=["H"],
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    frame_2 = Frame(
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbols=["He"],
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    dataset = Dataset(frames=[frame_1, frame_2])

    dataset.write(file_out=file_path_1, format=format, conditions=conditions_1)
    dataset.write(file_out=file_path_2, format=format, conditions=conditions_2)

    assert isfile(file_path_1)
    with open(file_path_1) as f:
        text = f.readlines()
        assert len(text) == lines
        assert "He" not in text

    assert isfile(file_path_2)
    with open(file_path_2) as f:
        text = f.readlines()
        assert len(text) == lines
        assert "He" not in text


def test_read_data_file_charge_warning(capsys: pytest.CaptureFixture):
    """
    Test we print a warning if the total charge is not equal to the sum of individual
    charges when reading from file, if verbosity is high enough.
    """
    Dataset(data_file="tests/data/n2p2/input.data", verbosity=1)

    total_charge = 0.055
    summed_charge = 0.06600000000000172
    assert capsys.readouterr().out == (
        f"WARNING: Total charge {total_charge} and sum of atomic charge {summed_charge}"
        " are not close\n"
    )
