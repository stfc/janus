"""
Utility functions for reading/writing to file, independent from `Data` or `ActiveLearning`
workflows.
"""

import re
from typing import Dict, Tuple

import numpy as np


def format_template_file(
    template_file: str,
    formatted_file: str,
    format_dict: Dict[str, str],
    format_shell_variables: bool = True,
):
    """"""
    with open(template_file) as f:
        template_text = f.read()

    if format_shell_variables:
        output_text = template_text
        for key, value in format_dict.items():
            # First format any shell variables ${} first, then any remaining {} patterns
            output_text = re.sub("\${" + key + "}", value, output_text)
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
