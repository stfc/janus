"""
Utility functions for reading/writing to file, independent from `Data` or `ActiveLearning`
workflows.
"""

import re
from shutil import copy
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

    return timesteps, extrapolation_free_lines, extrapolation_free_timesteps


def read_data_file(
    data_file: str,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Read n2p2 structure file and return the data as python objects.

    Parameters
    ----------
    data_file: str
        The complete n2p2 structure file path to read from.

    Returns
    -------
    list of tuple of `ndarray` and float
        Each tuple within the outer list represents a single frame from the structure file.
        Within the tuple, the elements are as follows:
          - ndarray of float with shape (3, 3) representing the lattice vectors
          - ndarray of str with length N, where N is the number of atoms in the frame,
            representing the chemical species of the atom
          - ndarray of float with shape (N, 3), where N is the number of atoms in the frame,
            representing the position vector of each atom
          - ndarray of float with shape (N, 3), where N is the number of atoms in the frame,
            representing the force vector of each atom
          - float of the frames total energy
    """
    with open(data_file) as f:
        lines = f.readlines()

    data = []
    lattice = []
    elements = []
    positions = []
    forces = []
    energy = None
    for line in lines:
        if line.strip() == "begin":
            lattice = []
            elements = []
            positions = []
            forces = []
            energy = None
        elif line.split()[0] == "lattice":
            lattice.append(np.array([float(line.split()[j]) for j in (1, 2, 3)]))
        elif line.split()[0] == "atom":
            elements.append(line.split()[4])
            positions.append(
                [
                    float(line.split()[1]),
                    float(line.split()[2]),
                    float(line.split()[3]),
                ]
            )
            forces.append(
                [
                    float(line.split()[-3]),
                    float(line.split()[-2]),
                    float(line.split()[-1]),
                ]
            )
        elif line.split()[0] == "energy":
            energy = float(line.split()[1])
        elif line.strip() == "end":
            data.append(
                (
                    np.array(lattice),
                    np.array(elements),
                    np.array(positions),
                    np.array(forces),
                    energy,
                )
            )

    return data


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
    dict of str and `ndarrays`
        Each key is the str of chemical species, with the value being a ndarray of float with
        length equal to the number of symmetry functions for that element.
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
                element = elements[int(data[0]) - 1]
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


# def read_atomenv_segments(
#     atomenv_name: str,
#     elements: List[str],
#     atoms_per_frame: int,
#     frame_segments: List[List[int]] = None,
# ):
#     """
#     TODO
#     """
#     element_environments = [{element: [] for element in elements} for _ in frame_segments]
#     with open(atomenv_name) as f:
#         atom_index = 0
#         frame_index = 0
#         line = f.readline()
#         while line:
#             for i, segment in enumerate(frame_segments):
#                 if frame_index in segment:
#                     segment_index = i

#             if atom_index == 0:
#                 for element in elements:
#                     element_environments[segment_index][element].append([])

#             data = line.split()
#             element_environments[segment_index][data[0]][-1].append(data[1:])

#             atom_index += 1
#             line = f.readline()

#             if atom_index == atoms_per_frame:
#                 atom_index = 0
#                 frame_index += 1

#     return [{key: np.array(value).astype(float) for key, value in segment.items()}
# for segment in element_environments]


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


def remove_data(
    remove_indices: List[int],
    data_file_in: str,
    data_file_out: str,
    data_file_backup: str,
):
    """
    Reads the content of `data_file_in`, removes the frames specified by `remove_indices`,
    then writes the remaining frames to `data_file_out`. If specified, a copy of the original
    data is made at `data_backup`.

    Parameters
    ----------
    data_file_in: str
        File path of the n2p2 structure file to read from.
    data_file_out: str
        File path of the n2p2 structure file to write to.
    data_file_backup: str
        File path of the n2p2 structure file to copy the original `data_file_in` to.
    """
    if len(remove_indices) == 0:
        print("No frames to remove")
        return

    if data_file_backup and data_file_in != data_file_backup:
        copy(data_file_in, data_file_backup)

    with open(data_file_in, "r") as f:
        lines = f.readlines()
    with open(data_file_out, "w") as f:
        i = 0
        for line in lines:
            if i not in remove_indices:
                f.write(line)

            if line.strip() == "end":
                i += 1
