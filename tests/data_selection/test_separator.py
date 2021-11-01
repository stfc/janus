"""
Unit tests for `separator.py`
"""

from os import listdir, remove
from os.path import isdir, isfile
from shutil import copy, rmtree
from typing import Dict, List, Literal, Union

import numpy as np
import pytest

from cc_hdnnp.data import Data
from cc_hdnnp.data_selection import Separator
from cc_hdnnp.structure import AllSpecies, AllStructures, Species, Structure


@pytest.fixture
def data():
    species = Species(symbol="H", atomic_number=1, mass=1.0)
    all_species = AllSpecies(species)
    structure = Structure(
        name="test", all_species=all_species, delta_E=1.0, delta_F=1.0
    )

    yield Data(
        structures=AllStructures(structure),
        main_directory="tests/data",
        n2p2_bin="",
        lammps_executable="",
    )

    for file in listdir("tests/data/tests_output"):
        if isfile("tests/data/tests_output/" + file):
            remove("tests/data/tests_output/" + file)
        elif isdir("tests/data/tests_output/" + file):
            rmtree("tests/data/tests_output/" + file)


@pytest.mark.parametrize("verbosity", [0, 1, 2])
@pytest.mark.parametrize(
    "criteria, differences",
    [
        (
            0.5,
            {
                "0[2]": 1.9442222,
                "1[2]": 0.6480741,
                "1[0]": 1.2961481,
                "1[0 2]": 0.6480741,
                "2[0]": 1.9442222,
            },
        ),
        (
            "mean",
            {
                "0[2]": 1.9502137,
                "1[2]": 0.6658329,
                "1[0]": 1.3051181,
                "1[0 2]": 0.6658329,
                "2[0]": 1.9502137,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "starting_frame_indices, settings, scaling, expected_starting_indices, "
    "expected_proposed_indices, expected_selected_indices, select_extreme_frames",
    [
        (
            None,
            "",
            "1 1 -1.0 -1.0 0.5 1.0\n1 2 -1.0 -1.0 1.0 2.0\n1 3 -1.0 -1.0 1.5 3.0\n",
            np.array([2]),
            np.array([1, 0]),
            np.array([2, 0]),
            False,
        ),
        (
            [2],
            "",
            "1 1 -1.0 -1.0 0.5 1.0\n1 2 -1.0 -1.0 1.0 2.0\n1 3 -1.0 -1.0 1.5 3.0\n",
            np.array([2]),
            np.array([1, 0]),
            np.array([2, 0]),
            False,
        ),
        (
            None,
            "",
            "1 1 0.0 1.0 0.5 1.0\n1 2 0.0 2.0 1.0 2.0\n1 3 0.0 3.0 1.5 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
        (
            [2],
            "",
            "1 1 0.0 1.0 0.5 1.0\n1 2 0.0 2.0 1.0 2.0\n1 3 0.0 3.0 1.5 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
        (
            None,
            "scale_symmetry_functions_sigma 1",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0]),
            np.array([2, 1]),
            np.array([0, 2]),
            True,
        ),
        (
            [2],
            "scale_symmetry_functions_sigma 1",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
        (
            None,
            "scale_symmetry_functions 1\nscale_min_short -0.3\nscale_max_short 0.3",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0]),
            np.array([2, 1]),
            np.array([0, 2]),
            True,
        ),
        (
            [2],
            "scale_symmetry_functions 1\nscale_min_short -0.3\nscale_max_short 0.3",
            "1 1 0.0 0.1 0.0 1.0\n1 2 0.0 0.4 0.0 2.0\n1 3 0.0 0.9 0.0 3.0\n",
            np.array([0, 2]),
            np.array([1]),
            np.array([0, 2, 1]),
            True,
        ),
    ],
)
def test_run_separation_selection(
    data: Data,
    capsys: pytest.CaptureFixture,
    settings: str,
    scaling: str,
    expected_starting_indices: List[int],
    expected_proposed_indices: List[int],
    expected_selected_indices: List[int],
    criteria: Union[float, Literal["mean"]],
    differences: Dict[str, float],
    starting_frame_indices: List[int],
    select_extreme_frames: bool,
    verbosity: int,
):
    """
    Test that the expected frame is removed, with the expected metric value(s),
    for a given input.
    """
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    copy("tests/data/n2p2/input.data.CUR", "tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.nn", "w") as f:
        f.write(settings)
    with open("tests/data/tests_output/scaling.data", "w") as f:
        f.write(scaling)
    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]
    difference = np.array(
        [
            differences["{0}{1}".format(i, expected_starting_indices)]
            for i in expected_proposed_indices
        ]
    )

    separator = Separator(data_controller=data, verbosity=verbosity)

    selected_indices = separator.run_separation_selection(
        n_frames_to_select=1,
        n_frames_to_compare=1,
        n_frames_to_propose=2,
        seed=0,
        criteria=criteria,
        starting_frame_indices=starting_frame_indices,
        select_extreme_frames=select_extreme_frames,
    )

    assert np.all(selected_indices == expected_selected_indices)

    text = capsys.readouterr().out
    if verbosity <= 0:
        assert "Values read from file in " not in text
        assert (
            "Starting separation selection with the following frames selected:\n{}\n".format(
                expected_starting_indices
            )
            not in text
        )
        assert "Proposed indices:\n{}\n".format(expected_proposed_indices) not in text
        assert (
            "Difference metric summed over all elements:\n{}\n".format(difference)
            not in text
        )
        assert (
            "Selected indices:\n{}\n".format(
                [expected_proposed_indices[np.argmax(difference)]]
            )
            not in text
        )
        assert "Time taken: " not in text
    else:
        assert "Values read from file in " in text
        assert (
            "Starting separation selection with the following frames selected:\n{}\n".format(
                expected_starting_indices
            )
            in text
        )
        if verbosity >= 2:
            assert "Proposed indices:\n{}\n".format(expected_proposed_indices) in text
            assert (
                "Difference metric summed over all elements:\n{}\n".format(difference)
                in text
            )
            assert (
                "Selected indices:\n{}\n".format(
                    [expected_proposed_indices[np.argmax(difference)]]
                )
                in text
            )
            assert "Time taken: " in text


@pytest.mark.parametrize(
    "criteria, settings, error",
    [
        (
            1.5,
            "",
            "`criteria` must be between 0 and 1, but was 1.5",
        ),
        (
            "mode",
            "",
            "`criteria` must be a quantile (float) between 0 and 1 or 'mean', but was mode",
        ),
        (
            0.5,
            "scale_symmetry_functions True\nscale_symmetry_functions_sigma True\n",
            "Both scale_symmetry_functions and scale_symmetry_functions_sigma "
            "were present in settings file.",
        ),
        (
            0.5,
            "scale_symmetry_functions True\n",
            "If scale_symmetry_functions is set, both scale_min_short and "
            "scale_max_short must be present.",
        ),
    ],
)
def test_run_separation_selection_errors(
    data: Data,
    criteria: Union[float, Literal["mean"]],
    settings: str,
    error: str,
):
    """
    Test that the expected errors are raised by giving the incorrect `critera` or `settings`.
    """
    copy("tests/data/n2p2/atomic-env.G", "tests/data/tests_output/atomic-env.G")
    copy("tests/data/n2p2/input.data", "tests/data/tests_output/input.data")
    with open("tests/data/tests_output/input.nn", "w") as f:
        f.write(settings)
    with open("tests/data/tests_output/scaling.data", "w") as f:
        f.write("1 1 0.0 1.0 0.5 1.0\n1 2 0.0 2.0 1.0 2.0\n1 3 0.0 3.0 1.5 3.0\n")

    data.n2p2_directories = ["tests/data/tests_output"]
    data.elements = ["H"]

    separator = Separator(data_controller=data)

    with pytest.raises(ValueError) as e:
        separator.run_separation_selection(
            n_frames_to_select=1,
            n_frames_to_compare=1,
            n_frames_to_propose=2,
            seed=0,
            criteria=criteria,
        )

    assert str(e.value) == error
