"""
Utility functions for reading/writing to file, independent from `Data` or `ActiveLearning`
workflows.
"""

import re
from typing import Dict, List, Tuple

import numpy as np


def format_template_file(
    template_file: str,
    formatted_file: str,
    format_dict: Dict[str, str],
    format_shell_variables: bool = True,
):
    """
    Uses the key value pairs in `format_dict` to format the text from `template_file` and write
    to `formatted_file`.

    Parameters
    ----------
    template_file: str
        The complete filepath to the text that will be formatted.
    formatted_file: str
        The complete filepath that the output is written to.
    format_dict: dict of str, str
        Each key value pair is a variable to format and its value.
    format_shell_variables: bool, optional
        If True, any shell variables marked as ${variable} will also be formatted.
        Default is True.
    """
    with open(template_file) as f:
        template_text = f.read()

    if format_shell_variables:
        output_text = template_text
        for key, value in format_dict.items():
            # First format any shell variables ${} first, then any remaining {} patterns
            output_text = re.sub(r"\${" + key + "}", value, output_text)
            output_text = re.sub("{" + key + "}", value, output_text)
    else:
        output_text = template_text.format(**format_dict)

    with open(formatted_file, "w") as f:
        f.write(output_text)


def read_lammps_log(
    dump_lammpstrj: int, log_lammps_file: str
) -> Tuple[np.ndarray, int, int]:
    """
    Reads a "log.lammps"and extracts information about if and at
    what timestep extrapolation of the network potential occured.

    Parameters
    ----------
    dump_lammpstrj : int
        Integer which defines that only every nth frame of the simulation is returned if no
        extrapolation occured.
    log_lammps_file : str
        The file path to the "log.lammps" file.

    Returns
    -------
    (np.ndarray, int, int)
        First element is array of int corresponding to timesteps, second is the number of
        extrapolation free lines and the third is the timestep that corresponds to that
        line.
    """
    with open(log_lammps_file) as f:
        data = [line for line in f.readlines()]

    if len(data) == 0:
        raise ValueError("{} was empty".format(log_lammps_file))

    # Count the number of lines that precede the simulation so they can be skipped
    counter = 0
    n_lines = len(data)
    while counter < n_lines and not data[counter].startswith("**********"):
        counter += 1

    # Starting at `counter`, check for extrapolation warnings
    extrapolation = False
    i = counter
    while i < n_lines and not (
        data[i].startswith("### NNP EXTRAPOLATION WARNING ###")
        # or data[i].startswith("### NNP EXTRAPOLATION SUMMARY ###")
    ):
        i += 1
    if i < n_lines:
        extrapolation = True
    i -= 1

    # The extrapolation warning (or end of simulation) look backwards to see how many steps
    # occured
    while i > counter and not data[i].startswith("thermo"):
        i -= 1
    if extrapolation:
        extrapolation_free_lines = i
        if i > counter:
            extrapolation_free_timesteps = int(data[i].split()[1])
        else:
            extrapolation_free_timesteps = -1
    else:
        extrapolation_free_lines = -1
        extrapolation_free_timesteps = int(data[i].split()[1])

    data = [
        int(line.split()[1]) if line.startswith("thermo") else -1
        for line in data[counter:]
        if line.startswith("thermo")
        or line.startswith("### NNP EXTRAPOLATION WARNING ###")
    ]

    # Subsample using `dump_lammpstrj`
    timesteps = np.unique(
        np.array(
            [
                data[i]
                for i in range(1, len(data))
                if data[i] != -1
                and (data[i] % dump_lammpstrj == 0 or data[i - 1] == -1)
            ]
        )
    )

    if len(timesteps) == 0:
        print("No timesteps completed for {}".format(log_lammps_file))
        timesteps = np.array([0])

    return timesteps, extrapolation_free_lines, extrapolation_free_timesteps


def read_normalisation(input_name: str) -> Tuple[float, float]:
    """
    Read the normalisation factors from the n2p2 settings file provided.

    Parameters
    ----------
    input_name : str
        The file path of the file to read normalisation from. Should be in a recognisable
        format, namely the "input.nn" file used for the network training.

    Returns
    -------
    float, float
        The conversion factor for energy and length respectively. If not found in the file,
        (1., 1.) is returned.
    """
    with open(input_name) as f:
        lines = f.readlines()[5:7]

    if lines[0].split()[0] == "conv_energy" and lines[1].split()[0] == "conv_length":
        return float(lines[0].split()[1]), float(lines[1].split()[1])

    return 1.0, 1.0


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


def read_nn_settings(
    settings_file: str, requested_settings: List[str]
) -> Dict[str, str]:
    """
    Read the settings of an n2p2 "input.nn" file and returns those in `requested_settings`.

    Parameters
    ----------
    settings_file: str
        The complete file path of an n2p2 "input.nn" file.
    requested_settings: list of str
        The list of settings keys to return.

    Returns
    -------
    dict of str and str
        Each key is the str of a setting name, with the value being the corresponding value.
        In general this may be a str, float, or int, so the returned value is kept as a str.
    """
    returned_settings = {}
    with open(settings_file) as f:
        lines = f.readlines()
        for setting in requested_settings:
            for line in lines:
                if line.strip() and line.split()[0] == setting:
                    returned_settings[setting] = line.split()[1]
                    break
            if setting not in returned_settings:
                print(
                    "Could not find setting {0} in {1}".format(setting, settings_file)
                )

    return returned_settings


def read_atomenv(
    atomenv_name: str, elements: List[str], atoms_per_frame: int, dtype: str = "float32"
) -> Dict[str, np.ndarray]:
    """


    Parameters
    ----------
    atomenv_name: str
        Complete file path of the n2p2 "atomic-env.G" file.
    elements: List[str]
        List of str representing the chemical symbol of all atoms in the structure.
    atoms_per_frame: int
        Number of atoms in each frame.
    dtype: str, optional
        numpy data type to use in the returned values. Default is "float32".

    Returns
    -------
    dict of str ndarray
        Each key is a str representing the chemical symbol of an element present in the
        structure. The corresponding value is an array of `dtype` with the shape (N, M, L)
        where N is the number of frames, M is the number of atoms per frame, and L is the
        number of symmetry functions.
    """
    element_environments = {element: [] for element in elements}
    with open(atomenv_name) as f:
        atom_index = 0
        frame_index = 0
        line = f.readline()
        while line:
            if atom_index == 0:
                for element in elements:
                    element_environments[element].append([])

            data = line.split()
            element_environments[data[0]][-1].append(data[1:])

            atom_index += 1
            line = f.readline()

            if atom_index == atoms_per_frame:
                atom_index = 0
                frame_index += 1

    return {
        key: np.array(value).astype(dtype)
        for key, value in element_environments.items()
    }


def read_last_timestep(file_lammps: str) -> int:
    """
    Attempts to read the last timestep which was written to `file_lammps`. Note that this
    may give an incorrect value if used on a file with a different format, as it assumes
    the timestep is the first integer on a given line.

    Parameters
    ----------
    file_lammps: str
        The complete file path of the LAMMPS log file to be read from.

    Returns
    -------
    int
        The last timestep which was written to `file_lammps`.
    """
    with open(file_lammps) as f:
        lines = f.readlines()
    # Start from the end of the file
    lines_reversed = reversed(lines)
    for line in lines_reversed:
        try:
            return int(line.split()[0])
        except IndexError:
            # If the line is blank, it does not correspond to a timestep
            pass
        except ValueError:
            # If the first word in the line cannot be cast to int,
            # it does not correspond to a timestep
            pass

    raise ValueError("Could not extract final timestep from {}".format(file_lammps))
