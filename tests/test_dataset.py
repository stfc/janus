"""
Unit tests for `structure.py`
"""

from typing import List

import numpy as np
import pytest

from cc_hdnnp.dataset import Dataset, Frame
from cc_hdnnp.structure import AllStructures, Species, Structure


def test_dataset():
    """
    Test that we can init a Dataset from file and check its properties.
    """
    dataset = Dataset(data_file="tests/data/n2p2/input.data")
    assert len(dataset) == 1
    assert dataset[0].lattice.shape == (3, 3)
    assert dataset[0].positions.shape == (512, 3)
    assert dataset[0].symbols.shape == (512,)
    assert dataset[0].forces.shape == (512, 3)
    assert dataset[0].energy != 0
    assert dataset[0].name is None


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
