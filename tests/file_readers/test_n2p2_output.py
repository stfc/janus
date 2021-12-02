"""
Unit tests for `file_readers/n2p2_output.py`
"""

import numpy as np
import pytest

from cc_hdnnp.file_readers import read_energies, read_forces


@pytest.mark.parametrize("format", ["point", "Conf.", "###"])
def test_read_n2p2_outputs(format: str):
    """
    Test that we read the energies and forces correctly for all formats.
    """
    energy_file = f"tests/data/n2p2/energy_{format}.out"
    forces_file = f"tests/data/n2p2/forces_{format}.out"
    np.testing.assert_array_equal(read_energies(energy_file), [[0, 2]])
    np.testing.assert_array_equal(read_forces(forces_file), [[0, 2], [0, 4], [0, 6]])


def test_read_n2p2_outputs_errors():
    """
    Test that we read the energies and forces correctly for all formats.
    """
    file = "tests/data/n2p2/input.data"
    with pytest.raises(OSError) as e:
        read_energies(file)
    assert str(e.value) == "Unknown RuNNer format"
    with pytest.raises(OSError) as e:
        read_forces(file)
    assert str(e.value) == "Unknown RuNNer format"
