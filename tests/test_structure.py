"""
Unit tests for `structure.py`
"""

import pytest

from cc_hdnnp.structure import AllStructures, Species, Structure


def test_all_structures_same_name():
    """
    Test that creating an AllStructures object with multiple structures with the same name
    raises an error.
    """
    name = "test"
    species = Species(symbol="H", atomic_number=1, mass=1.0)
    structure = Structure(name=name, all_species=[species], delta_E=None, delta_F=None)
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
    structure_1 = Structure(
        name="test_1", all_species=[species_1], delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=[species_2], delta_E=None, delta_F=None
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
    species_2 = Species(symbol="H", atomic_number=1, mass=4.0)
    # Bypass the validation in Species.__init__
    species_2.atomic_number = 2
    structure_1 = Structure(
        name="test_1", all_species=[species_1], delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=[species_2], delta_E=None, delta_F=None
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
    structure_1 = Structure(
        name="test_1", all_species=[species_1], delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=[species_2], delta_E=None, delta_F=None
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
    structure_1 = Structure(
        name="test_1", all_species=[species_1], delta_E=None, delta_F=None
    )
    structure_2 = Structure(
        name="test_2", all_species=[species_2], delta_E=None, delta_F=None
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
    with pytest.raises(ValueError) as e:
        Structure(
            name="test_1",
            all_species=[species_1],
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
    with pytest.raises(TypeError) as e:
        Structure(
            name="test_1",
            all_species=[species_1],
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
    with pytest.raises(ValueError) as e:
        Structure(
            name="test_1",
            all_species=[species_1],
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
    structure = Structure(
        name="test_1",
        all_species=[species_1],
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
    with pytest.raises(TypeError) as e:
        Structure(
            name="test_1",
            all_species=[species_1],
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
    with pytest.raises(ValueError) as e:
        Structure(
            name="test_1",
            all_species=[species_1],
            delta_E=None,
            delta_F=None,
            max_interpolated_structures_per_simulation=max_interpolated_structures_per_s,
        )
    assert (
        str(e.value)
        == "`max_interpolated_structures_per_simulation` must have a value of at least 0"
    )


@pytest.mark.parametrize(
    "symbol, atomic_number, valence",
    [
        ("H", 1, None),
        ("H", None, None),
        (None, 1, None),
        ("H", 1, 1),
    ],
)
def test_species(symbol: str, atomic_number: int, valence: int):
    """
    Test that 1 or both of `symbol` or `atomic_number` leads to both being set, along with
    mass.
    """
    species = Species(symbol=symbol, atomic_number=atomic_number, valence=valence)
    assert isinstance(species.symbol, str)
    assert isinstance(species.atomic_number, int)
    assert isinstance(species.mass, float)
    assert species.valence == valence


@pytest.mark.parametrize(
    "symbol, atomic_number, valence, error",
    [
        ("H", 2, None, "Provided symbol H does not match provided atomic number 2"),
        (
            None,
            None,
            None,
            "At least one of `symbol` or `atomic_number` must be provided.`",
        ),
        ("H", None, -1, "`valence` must not be negative, but was -1."),
        (
            "H",
            None,
            10,
            "`valence` cannot be greater than the `atomic_number`, but they were 10, 1.",
        ),
    ],
)
def test_species_error(symbol: str, atomic_number: int, valence: int, error: str):
    """
    Test that providing neither `symbol` nor `atomic_number`, or a pair that don't match,
    leads to a ValueError.
    """
    with pytest.raises(ValueError) as e:
        Species(symbol=symbol, atomic_number=atomic_number, valence=valence)
    assert str(e.value) == error


def test_structure_get_species_failure():
    """
    Test that `Structure.get_species` raises an error when given a symbol that is not present.
    """
    symbol = "Li"
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="He", atomic_number=2, mass=4.0)
    structure = Structure(
        name="test", all_species=[species_1, species_2], delta_E=0.0, delta_F=0.0
    )
    with pytest.raises(ValueError) as e:
        structure.get_species(symbol)
    assert str(
        e.value
    ) == "No atomic species with symbol `{0}` present in `{1}`." "".format(
        symbol, structure.element_list
    )


def test_structure_get_species_success():
    """
    Test that `Structure.get_species` succeeds.
    """
    symbol = "H"
    species_1 = Species(symbol="H", atomic_number=1, mass=1.0)
    species_2 = Species(symbol="He", atomic_number=2, mass=4.0)
    structure = Structure(
        name="test", all_species=[species_1, species_2], delta_E=0.0, delta_F=0.0
    )
    test_species = structure.get_species(symbol)
    assert test_species == species_1


def test_structure_min_separation_default():
    """
    Test that `Structure` sets the default values for `min_separation` correctly.
    """
    species_1 = Species(
        symbol="H", atomic_number=1, mass=1.0, min_separation={"H": 1.0}
    )
    species_2 = Species(
        symbol="He", atomic_number=2, mass=4.0, min_separation={"H": 2.0}
    )
    all_species = Structure(
        name="test",
        all_species=[species_1, species_2],
        global_separation=3.0,
        delta_E=0.0,
        delta_F=0.0,
    )
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
