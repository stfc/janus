"""
Unit tests for `data_operations.py`
"""
import numpy as np
import pytest

from cc_hdnnp.data_operations import check_nearest_neighbours


@pytest.mark.parametrize(
    "pos_i, pos_j",
    [
        (np.array([]), np.array([])),
        (np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])),
    ],
)
def test_check_nearest_neighbours(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
):
    """
    Test that we accept the neighbours in the cases where they are empty or only have 1 dimension.
    """
    accepted, d = check_nearest_neighbours(
        lat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        pos_i=pos_i,
        pos_j=pos_j,
        ii=False,
        d_min=0.1,
    )

    assert accepted
    assert d == -1
