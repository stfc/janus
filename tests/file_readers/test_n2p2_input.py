"""
Unit tests for `file_readers/n2p2_input.py`
"""
from janus.file_readers import read_nn_settings


def test_read_nn_settings_normalisation_defaults():
    """Test that we return default values if normalisation header cannot be found."""
    settings_file = "tests/data/n2p2/input.nn"
    expected_settings = {"mean_energy": "0", "conv_energy": "1", "conv_length": "1"}
    returned_settings = read_nn_settings(
        settings_file=settings_file, requested_settings=list(expected_settings.keys())
    )

    assert returned_settings == expected_settings
