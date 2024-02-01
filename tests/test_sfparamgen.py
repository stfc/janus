"""
Unit tests for `sfparamgen.py`
"""

import warnings

import numpy as np
import pytest

from janus.sfparamgen import SymFuncParamGenerator


@pytest.fixture
def mock_warn(mocker):
    """Mocks `warnings.warn`."""
    mocker.patch("warnings.warn")


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
    Test not passing `r_lower` for "gastegger2018" raises an error.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=1.0)

    with pytest.raises(TypeError) as e:
        generator.generate_radial_params(
            rule="gastegger2018", mode="", nb_param_pairs=2
        )
    assert str(e.value) == 'Argument r_lower is required for rule "gastegger2018"'


def test_r_lower_center_error():
    """
    Test passing `r_lower=0` for "center" "gastegger2018" raises an error.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=1.0)

    with pytest.raises(ValueError) as e:
        generator.generate_radial_params(
            rule="gastegger2018", mode="center", nb_param_pairs=2, r_lower=0
        )
    assert str(e.value) == (
        "Invalid argument(s): rule = gastegger2018, "
        "mode = center requires that 0 < "
        "r_lower < r_upper <= r_cutoff."
    )


def test_r_lower_shift_error():
    """
    Test passing `r_lower<0` for "shift" "gastegger2018" raises an error.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=1.0)

    with pytest.raises(ValueError) as e:
        generator.generate_radial_params(
            rule="gastegger2018", mode="shift", nb_param_pairs=2, r_lower=-1
        )
    assert str(e.value) == (
        "Invalid argument(s): rule = gastegger2018, "
        "mode = shift requires that 0 <= "
        "r_lower < r_upper <= r_cutoff."
    )


def test_mode_error():
    """
    Test passing an unrecognised mode raises an error.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=1.0)

    with pytest.raises(ValueError) as e:
        generator.generate_radial_params(
            rule="gastegger2018", mode="unrecognised", nb_param_pairs=2, r_lower=1.0
        )
    assert str(e.value) == 'invalid argument for "mode"'

    with pytest.raises(ValueError) as e:
        generator.generate_radial_params(
            rule="imbalzano2018", mode="unrecognised", nb_param_pairs=2
        )
    assert str(e.value) == 'invalid argument for "mode"'


def test_rule_error():
    """
    Test passing an unrecognised rule raises an error.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=1.0)

    with pytest.raises(ValueError) as e:
        generator.generate_radial_params(
            rule="unrecognised", mode="unrecognised", nb_param_pairs=2
        )
    assert str(e.value) == 'invalid argument for "rule"'


def test_gastegger_center():
    """
    Test r_shift_grid and eta_grid for gastegger center.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)

    generator.generate_radial_params(
        rule="gastegger2018", mode="center", nb_param_pairs=2, r_lower=0.5, r_upper=1.0
    )
    assert all(generator._r_shift_grid == np.array([0.0, 0.0]))
    assert all(generator._eta_grid == np.array([2.0, 0.5]))


def test_gastegger_shift():
    """
    Test r_shift_grid and eta_grid for gastegger shift.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)

    generator.generate_radial_params(
        rule="gastegger2018", mode="shift", nb_param_pairs=2, r_lower=0.0, r_upper=1.0
    )
    assert all(generator._r_shift_grid == np.array([0.0, 1.0]))
    assert all(generator._eta_grid == np.array([0.5, 0.5]))


def test_imbalzano_warning_lower(mock_warn):
    """
    Test warnings are given when redundant arguments are passed.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)

    generator.generate_radial_params(
        rule="imbalzano2018", mode="shift", nb_param_pairs=2, r_lower=0.0
    )
    warnings.warn.assert_any_call(
        "The argument r_lower to method generate_radial_params will be ignored,"
        ' since it is unused when calling the method with rule="imbalzano2018".'
    )


def test_imbalzano_warning_upper(mock_warn):
    """
    Test warnings are given when redundant arguments are passed.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)

    generator.generate_radial_params(
        rule="imbalzano2018", mode="shift", nb_param_pairs=2, r_upper=1.0
    )
    warnings.warn.assert_any_call(
        "The argument r_upper to method"
        " generate_radial_params will be ignored,"
        " since it is unused when calling the method"
        ' with rule="imbalzano2018".'
    )


def test_imbalzano_shift():
    """
    Test r_shift_grid and eta_grid for imbalzano shift.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)

    generator.generate_radial_params(
        rule="imbalzano2018", mode="shift", nb_param_pairs=2
    )
    np.testing.assert_allclose(generator._r_shift_grid, np.array([1.0, 2 ** 0.5]))
    np.testing.assert_allclose(
        generator._eta_grid, np.array([3 + 2 ** (3 / 2), 3 / 2 + 2 ** (1 / 2)])
    )


def test_imbalzano_center():
    """
    Test r_shift_grid and eta_grid for imbalzano center.
    """
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)

    generator.generate_radial_params(
        rule="imbalzano2018", mode="center", nb_param_pairs=2
    )
    assert all(generator._r_shift_grid == np.array([0.0, 0.0]))
    assert all(generator._eta_grid == np.array([0.25, 0.25]))


def test_set_custom_radial_params_length():
    """Test that passing values of different lengths raises an error."""
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)
    with pytest.raises(TypeError) as e:
        generator.set_custom_radial_params([1], [2, 2])

    assert str(e.value) == "r_shift_values and eta_values must have same length."


def test_set_custom_radial_params_r_shift():
    """Test that passing negative r_shift raises an error."""
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)
    with pytest.raises(ValueError) as e:
        generator.set_custom_radial_params([-1, 1], [2, 2])

    assert str(e.value) == "r_shift_values must all be non-negative."


def test_set_custom_radial_params_eta():
    """Test that passing negative values of eta raises an error."""
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)
    with pytest.raises(ValueError) as e:
        generator.set_custom_radial_params([1, 1], [-2, 2])

    assert str(e.value) == "eta_values must all be greater than zero."


def test_set_custom_radial_params_success():
    """Test setting values successfully."""
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)
    generator.set_custom_radial_params([1, 1], [2, 2])

    assert generator.radial_paramgen_settings is None
    assert all(generator.r_shift_grid == np.array([1, 1]))
    assert all(generator.eta_grid == np.array([2, 2]))


def test_write_settings_overview(capsys: pytest.CaptureFixture):
    """Test writing settings overview to stdout."""
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)
    generator.symfunc_type = "radial"
    generator.generate_radial_params(
        rule="gastegger2018", mode="shift", nb_param_pairs=2, r_lower=0.0, r_upper=1.0
    )
    generator.write_settings_overview()

    assert capsys.readouterr().out == (
        "#########################################################################\n"
        "# Radial symmetry function set, for elements []\n"
        "#########################################################################\n"
        "# r_cutoff       = 2.0\n"
        "# The following settings were used for generating sets\n"
        "# of values for the radial parameters r_shift and eta:\n"
        "# rule           = gastegger2018\n"
        "# mode           = shift\n"
        "# nb_param_pairs = 2\n"
        "# r_lower        = 0.0\n"
        "# r_upper        = 1.0\n"
        "# Sets of values for parameters:\n"
        "# r_shift_grid   = [0. 1.]\n"
        "# eta_grid       = [0.5 0.5]\n"
        "\n"
    )


def test_write_settings_overview_custom(capsys: pytest.CaptureFixture):
    """Test writing settings overview to stdout after using custom values."""
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)
    generator.symfunc_type = "radial"
    generator.set_custom_radial_params([1, 1], [2, 2])
    generator.write_settings_overview()

    assert capsys.readouterr().out == (
        "#########################################################################\n"
        "# Radial symmetry function set, for elements []\n"
        "#########################################################################\n"
        "# r_cutoff       = 2.0\n"
        "# A custom set of values was used for the radial parameters r_shift and eta.\n"
        "# Thus, there are no settings on radial parameter generation available for display.\n"
        "# Sets of values for parameters:\n"
        "# r_shift_grid   = [1 1]\n"
        "# eta_grid       = [2 2]\n"
        "\n"
    )


def test_write_settings_overview_angular(capsys: pytest.CaptureFixture):
    """Test writing settings overview to stdout with angular functions."""
    generator = SymFuncParamGenerator(elements=[], r_cutoff=2.0)
    generator.symfunc_type = "angular_narrow"
    generator.zetas = [1]
    generator.set_custom_radial_params([1, 1], [2, 2])
    generator.write_settings_overview()

    assert capsys.readouterr().out == (
        "#########################################################################\n"
        "# Narrow angular symmetry function set, for elements []\n"
        "#########################################################################\n"
        "# r_cutoff       = 2.0\n"
        "# A custom set of values was used for the radial parameters r_shift and eta.\n"
        "# Thus, there are no settings on radial parameter generation available for display.\n"
        "# Sets of values for parameters:\n"
        "# r_shift_grid   = [1 1]\n"
        "# eta_grid       = [2 2]\n"
        "# lambdas        = [-1.  1.]\n"
        "# zetas          = [1]\n"
        "\n"
    )


def test_write_parameter_strings_angular(capsys: pytest.CaptureFixture):
    """Test writing parameter strings to stdout with angular functions."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    generator.symfunc_type = "angular_narrow"
    generator.zetas = [1]
    generator.set_custom_radial_params([1, 1], [2, 2])
    generator.write_parameter_strings()

    assert capsys.readouterr().out == (
        "symfunction_short H  3 H  H  2.000E+00 -1 1.000E+00 2.000E+00 1.000E+00\n"
        "symfunction_short H  3 H  H  2.000E+00  1 1.000E+00 2.000E+00 1.000E+00\n"
        "symfunction_short H  3 H  H  2.000E+00 -1 1.000E+00 2.000E+00 1.000E+00\n"
        "symfunction_short H  3 H  H  2.000E+00  1 1.000E+00 2.000E+00 1.000E+00\n"
        "\n"
    )


def test_write_parameter_weighted_radial(capsys: pytest.CaptureFixture):
    """Test writing parameter strings to stdout with weighted radial functions."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    generator.symfunc_type = "weighted_radial"
    generator.zetas = [1]
    generator.set_custom_radial_params([1, 1], [2, 2])
    generator.write_parameter_strings()

    assert capsys.readouterr().out == (
        "symfunction_short H  12 2.000E+00 1.000E+00 2.000E+00\n"
        "symfunction_short H  12 2.000E+00 1.000E+00 2.000E+00\n"
        "\n"
    )


def test_write_parameter_strings_weighted_angular(capsys: pytest.CaptureFixture):
    """Test writing parameter strings to stdout with weighted_angular functions."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    generator.symfunc_type = "weighted_angular"
    generator.zetas = [1]
    generator.set_custom_radial_params([1, 1], [2, 2])
    generator.write_parameter_strings()

    assert capsys.readouterr().out == (
        "symfunction_short H  13 2.000E+00 1.000E+00 -1 1.000E+00 2.000E+00\n"
        "symfunction_short H  13 2.000E+00 1.000E+00  1 1.000E+00 2.000E+00\n"
        "symfunction_short H  13 2.000E+00 1.000E+00 -1 1.000E+00 2.000E+00\n"
        "symfunction_short H  13 2.000E+00 1.000E+00  1 1.000E+00 2.000E+00\n"
        "\n"
    )


def test_check_writing_prerequisites_r_shift_eta():
    """Test an error is raised when r_shift or eta not set."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    generator.symfunc_type = "radial"
    with pytest.raises(ValueError) as e:
        generator.check_writing_prerequisites()

    assert str(e.value) == "Values for r_shift and/or eta not set."


def test_check_writing_prerequisites_zeta():
    """Test an error is raised when zeta not set."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    generator.symfunc_type = "angular_narrow"
    generator.set_custom_radial_params([1, 1], [2, 2])
    with pytest.raises(ValueError) as e:
        generator.check_writing_prerequisites()

    assert str(e.value) == (
        "Values for zeta not set (required for symmetry function type angular_narrow).\n"
        " If you are seeing this error despite having previously set zetas, make sure\n"
        " they have not been cleared since by setting a non-angular symmetry function type."
    )


def test_check_writing_prerequisites_r_shift_eta_calling_method():
    """Test an error is raised when r_shift or eta not set with calling method."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    generator.symfunc_type = "radial"
    with pytest.raises(ValueError) as e:
        generator.check_writing_prerequisites("method")

    assert str(e.value) == (
        "Values for r_shift and/or eta not set. "
        "Calling method method requires that values for r_shift and eta have been set before."
    )


def test_check_writing_prerequisites_zeta_calling_method():
    """Test an error is raised when zeta not set with calling method."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    generator.symfunc_type = "angular_narrow"
    generator.set_custom_radial_params([1, 1], [2, 2])
    with pytest.raises(ValueError) as e:
        generator.check_writing_prerequisites("method")

    assert str(e.value) == (
        "Values for zeta not set.\n "
        "Calling method, while using symmetry function type angular_narrow,\n"
        " requires zetas to have been set before.\n "
        "If you are seeing this error despite having previously set zetas, make sure\n"
        " they have not been cleared since by setting a non-angular symmetry function type."
    )


def test_check_symfunc_type():
    """Test an error is raised type not set."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    with pytest.raises(ValueError) as e:
        generator.check_symfunc_type()

    assert str(e.value) == "Symmetry function type not set."


def test_check_symfunc_type_method():
    """Test an error is raised type not set with a calling method."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    with pytest.raises(ValueError) as e:
        generator.check_symfunc_type("method")

    assert str(e.value) == (
        "Symmetry function type not set. "
        "Calling method method requires that symmetry function type have been set before."
    )


def test_set_bad_type():
    """Test an error is raised when setting an unrecognised type."""
    generator = SymFuncParamGenerator(elements=["H"], r_cutoff=2.0)
    with pytest.raises(ValueError) as e:
        generator.symfunc_type = "type"

    assert str(e.value) == (
        "Invalid symmetry function type. Must be one of "
        "['radial', 'angular_narrow', 'angular_wide', 'weighted_radial', 'weighted_angular']"
    )
