"""
Reads n2p2 scaling.data files for information used in the workflow.
"""

from typing import Dict, List


def read_scaling(
    scaling_file: str, elements: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Read the min/max ranges of an n2p2 "scaling.data" file for all elements and symmetry
    functions.

    Parameters
    ----------
    scaling_file: str
        The complete file path of an n2p2 "scaling.data" file.
    elements: list of str
        The list of chemical symbols ordered by atomic number.

    Returns
    -------
    Dict[str, Dict[str, List[float]]]
        Each key is the str of chemical species, with the value being another dictionary.
        Here the keys are {"min", "max", "mean", "sigma"}, and the value ndarray of float with
        length equal to the number of symmetry functions for the element.
    """
    element_ranges = {
        element: {
            "min": [],
            "max": [],
            "mean": [],
            "sigma": [],
        }
        for element in elements
    }
    with open(scaling_file) as f:
        line = f.readline()
        while line:
            if not line.startswith("#"):
                data = line.split()
                elment_index = int(data[0])
                element = elements[elment_index - 1]
                element_ranges[element]["min"].append(float(data[2]))
                element_ranges[element]["max"].append(float(data[3]))
                element_ranges[element]["mean"].append(float(data[4]))
                element_ranges[element]["sigma"].append(float(data[5]))

            line = f.readline()

    return element_ranges
