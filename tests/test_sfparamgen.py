"""
Unit tests for `sfparamgen.py`
"""

import pytest

from cc_hdnnp.sfparamgen import SymFuncParamGenerator


def test_r_cutoff_error():
    """
    Test passing a negative r_cutoff raises an error.
    """
    with pytest.raises(ValueError) as e:
        SymFuncParamGenerator(elements=[], r_cutoff=-1.0)
    assert str(e.value) == "Invalid cutoff radius given. Must be greater than zero."


def test_nb_param_pairs_error():
    """
    Test passing `nb_param_pairs` less than 2 raises an error.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=1.0)

    with pytest.raises(ValueError) as e:
        generator.generate_radial_params(rule="", mode="", nb_param_pairs=1)
    assert str(e.value) == "nb_param_pairs must be two or greater."


def test_r_lower_error():
    """
    Test passing `nb_param_pairs` less than 2 raises an error.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=1.0)

    with pytest.raises(TypeError) as e:
        generator.generate_radial_params(
            rule="gastegger2018", mode="", nb_param_pairs=2
        )
    assert str(e.value) == 'Argument r_lower is required for rule "gastegger2018"'
