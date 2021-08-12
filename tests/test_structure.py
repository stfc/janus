"""
Unit tests for `structure.py`
"""

import pytest

from cc_hdnnp.structure import AllSpecies, AllStructures, Species, Structure


def test_all_structures_same_name():
    """
    Test that creating an AllStructures object with multiple structures with the same name
    raises an error.
    """
    name = "test"
    species = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species = AllSpecies(species)
    structure = Structure(
        name=name, all_species=all_species, delta_E=None, delta_F=None
    )
    with pytest.raises(ValueError) as e:
        AllStructures(structure, structure)
    assert str(e.value) == "Cannot have multiple structures with the name `{}`".format(
        name
    )


def test_all_structures_element_list():
    """
    Test that creating an AllStructures object with multiple structures with different elements
    raises an error.
    """
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="He", atomic_number=2, mass=4.0)
    all_species_1 = AllSpecies(species_1)
    all_species_2 = AllSpecies(species_2)
    structure_1 = Structure(
        name="test_1", all_species=all_species_1, delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=all_species_2, delta_E=None, delta_F=None
    )
    with pytest.raises(ValueError) as e:
        AllStructures(structure_1, structure_2)
    assert str(e.value) == "All structures must have the same `element_list`"


def test_all_structures_atomic_number_list():
    """
    Test that creating an AllStructures object with multiple structures with different atomic
    numbers raises an error.
    """
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="H", atomic_number=2, mass=4.0)
    all_species_1 = AllSpecies(species_1)
    all_species_2 = AllSpecies(species_2)
    structure_1 = Structure(
        name="test_1", all_species=all_species_1, delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=all_species_2, delta_E=None, delta_F=None
    )
    with pytest.raises(ValueError) as e:
        AllStructures(structure_1, structure_2)
    assert str(e.value) == "All structures must have the same `atomic_number_list`"


def test_all_structures_mass_list():
    """
    Test that creating an AllStructures object with multiple structures with different masses
    raises an error.
    """
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="H", atomic_number=1, mass=4.0)
    all_species_1 = AllSpecies(species_1)
    all_species_2 = AllSpecies(species_2)
    structure_1 = Structure(
        name="test_1", all_species=all_species_1, delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=all_species_2, delta_E=None, delta_F=None
    )
    with pytest.raises(ValueError) as e:
        AllStructures(structure_1, structure_2)
    assert str(e.value) == "All structures must have the same `mass_list`"


def test_all_structures_success():
    """
    Test creating an AllStructures object successfully.
    """
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species_1 = AllSpecies(species_1)
    all_species_2 = AllSpecies(species_2)
    structure_1 = Structure(
        name="test_1", all_species=all_species_1, delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=all_species_2, delta_E=None, delta_F=None
    )
    all_structures = AllStructures(structure_1, structure_2)
    assert all_structures.element_list == ["H"]
    assert all_structures.mass_list == [1.0]


def test_structure_selection_length():
    """
    Test that creating a Structure with a selection with more than 2 entries raises an error.
    """
    selection = [0, 1, 2]
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species_1 = AllSpecies(species_1)
    with pytest.raises(ValueError) as e:
        Structure(
            name="test_1",
            all_species=all_species_1,
            delta_E=None,
            delta_F=None,
            selection=selection,
        )
    assert str(e.value) == "`selection` must have 2 entries, but was {}".format(
        selection
    )


def test_structure_selection_type():
    """
    Test that creating a Structure with a selection with the wrong type raises an error.
    """
    selection = ["0", 1.0]
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species_1 = AllSpecies(species_1)
    with pytest.raises(TypeError) as e:
        Structure(
            name="test_1",
            all_species=all_species_1,
            delta_E=None,
            delta_F=None,
            selection=selection,
        )
    assert str(
        e.value
    ) == "`selection` entries must have type `int`, but were of type " "{0} and {1}".format(
        type(selection[0]), type(selection[1])
    )


def test_structure_selection_value():
    """
    Test that creating a Structure with a selection with the wrong values raises an error.
    """
    selection = [-1, 0]
    msg = (
        "`selection` entries must be at least 0 and 1 respectively, but were "
        "{0} and {1}"
    )
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species_1 = AllSpecies(species_1)
    with pytest.raises(ValueError) as e:
        Structure(
            name="test_1",
            all_species=all_species_1,
            delta_E=None,
            delta_F=None,
            selection=selection,
        )
    assert str(e.value) == msg.format(selection[0], selection[1])


def test_structure_selection_success():
    """
    Test creating a Structure with a selection successfully.
    """
    selection = [1, 10]
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species_1 = AllSpecies(species_1)
    structure = Structure(
        name="test_1",
        all_species=all_species_1,
        delta_E=None,
        delta_F=None,
        selection=selection,
    )
    assert structure.selection == selection


def test_structure_max_interpolated_type():
    """
    Test that creating a Structure with a `max_interpolated_structures_per_simulation` with
    the wrong type raises an error.
    """
    max_interpolated_structures_per_s = "-1"
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species_1 = AllSpecies(species_1)
    with pytest.raises(TypeError) as e:
        Structure(
            name="test_1",
            all_species=all_species_1,
            delta_E=None,
            delta_F=None,
            max_interpolated_structures_per_simulation=max_interpolated_structures_per_s,
        )
    assert (
        str(e.value)
        == "`max_interpolated_structures_per_simulation` must have type `int`"
    )


def test_structure_max_interpolated_value():
    """
    Test that creating a Structure with a `max_interpolated_structures_per_simulation`
    with the wrong value raises an error.
    """
    max_interpolated_structures_per_s = -1
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species_1 = AllSpecies(species_1)
    with pytest.raises(ValueError) as e:
        Structure(
            name="test_1",
            all_species=all_species_1,
            delta_E=None,
            delta_F=None,
            max_interpolated_structures_per_simulation=max_interpolated_structures_per_s,
        )
    assert (
        str(e.value)
        == "`max_interpolated_structures_per_simulation` must have a value of at least 0"
    )


def test_all_species_get_species_failure():
    """
    Test that `AllSpecies.get_species` raises an error when given a symbol that is not present.
    """
    symbol = "Li"
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="He", atomic_number=2, mass=4.0)
    all_species = AllSpecies(species_1, species_2)
    with pytest.raises(ValueError) as e:
        all_species.get_species(symbol)
    assert str(
        e.value
    ) == "No atomic species with symbol `{0}` present in `{1}`." "".format(
        symbol, all_species.element_list
    )


def test_all_species_get_species_success():
    """
    Test that `AllSpecies.get_species` succeeds.
    """
    symbol = "H"
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="He", atomic_number=2, mass=4.0)
    all_species = AllSpecies(species_1, species_2)
    test_species = all_species.get_species(symbol)
    assert test_species == species_1


def test_all_species_min_separation_default():
    """
    Test that `AllSpecies` sets the default values for `min_separation` correctly.
    """
    species_1 = Species(
        symbol="H", atomic_number=1, mass=1.0, min_separation={"H": 1.0}
    )
    species_2 = Species(
        symbol="He", atomic_number=2, mass=4.0, min_separation={"H": 2.0}
    )
    all_species = AllSpecies(species_1, species_2, global_separation=3.0)
    test_species_1 = all_species.get_species("H")
    test_species_2 = all_species.get_species("He")
    assert test_species_1.min_separation["H"] == 1.0
    assert test_species_1.min_separation["He"] == 3.0
    assert test_species_2.min_separation["H"] == 2.0
    assert test_species_2.min_separation["He"] == 3.0


def test_all_structures_empty():
    """
    Test that `AllStructures` raises an error when no arguments passed.
    """
    with pytest.raises(ValueError) as e:
        AllStructures()
    assert str(e.value) == "At least one `Structure` object must be passed."
