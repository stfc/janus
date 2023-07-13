"""
Reads lammps.log files for information used in the workflow.
"""

from typing import Tuple, List, Dict

import numpy as np

def read_lammps_log(
    dump_lammpstrj: int,
    log_lammps_file: str,
    output_headers: List[str] = ["Temp"],
) -> Tuple[np.ndarray, int, int, Dict[str, np.ndarray]]:
    """
    Reads a "log.lammps" and extracts information about if and at
    what timestep extrapolation of the network potential occurred.

    Parameters
    ----------
    dump_lammpstrj : int
        Integer which defines that only every nth frame of the simulation is returned if no
        extrapolation occurred.
    log_lammps_file : str
        The file path to the "log.lammps" file.
    output_headers: List[str] = ["Temp"]
        List of log headers for variable(s) to return. Default is `["Temp"]`.

    Returns
    -------
    (np.ndarray, int, int, Dict[str, np.ndarray])
        First element is array of int corresponding to timesteps, second is the number of
        extrapolation free lines and the third is the timestep that corresponds to that
        line. The fourth is a dictionary, with log headers as keys, and header values
        at each timestep of the simulation as dictionary values.
    """
    with open(log_lammps_file) as f:
        data = f.readlines()

    if len(data) == 0:
        raise ValueError("{} was empty".format(log_lammps_file))

    # Count the number of lines that precede the simulation so they can be skipped
    step_index = 0

    header_line_number = 0
    n_lines = len(data)
    while header_line_number < n_lines and "Step" not in data[header_line_number]:
        header_line_number += 1

    if header_line_number == n_lines:
        raise ValueError(
            f"{log_lammps_file} does not contain simulation headers, "
            "indicating it crashed before the 0th timestep."
        )
    headers = data[header_line_number].split()
    step_index = headers.index("Step")

    header_indices = [headers.index(output_header) for output_header in output_headers]
    output_list = np.array([[]])
    timesteps_list = []
    accept_next = False
    extrapolation = False
    i = header_line_number + 1

    # Starting at `header_line_number + 1`, check for extrapolation warnings
    while i < n_lines:
        line = data[i]
        if not extrapolation and (
            data[i].startswith("### NNP EXTRAPOLATION WARNING ###")
            or (
                data[i].startswith("### NNP EXTRAPOLATION SUMMARY ###")
                and int(data[i].split()[8]) != 0
            )
            or (
                data[i].startswith("### NNP EW SUMMARY ###")
                and int(data[i].split()[8]) != 0
            )
            or ("Too many extrapolation warnings" in data[i])
        ):
            # At the first sign of extrapolation, set `extrapolation_free_lines`
            # and `extrapolation_free_timesteps`
            extrapolation = True
            extrapolation_free_lines = i
            if i > header_line_number:
                extrapolation_free_timesteps = timesteps_list[-1]
            else:
                # Should not be able to reach this, as 0th timestep is written immediately
                # after the headers, so assign -1
                extrapolation_free_timesteps = -1
        else:
            try:
                for header_index in header_indices:
                    output_list = np.append(
                        output_list,
                        np.array([float(line.split()[header_index])])
                    )
                timestep = int(line.split()[step_index])
                if accept_next or timestep % dump_lammpstrj == 0:
                    timesteps_list.append(timestep)
                    accept_next = False
            except ValueError:
                if line.startswith("Loop time of "):
                    while i < n_lines and "Step" not in data[i]:
                        i += 1
                    if i == n_lines:
                        break
                    else:
                        i += 1
                elif (
                    line.startswith("### NNP EXTRAPOLATION WARNING ###")
                    or (
                        line.startswith("### NNP EXTRAPOLATION SUMMARY ###")
                        and int(line.split()[8]) != 0
                    )
                    or (
                        line.startswith("### NNP EW SUMMARY ###")
                        and int(line.split()[8]) != 0
                    )
                ):
                    # If we cannot cast to int due to an EW, then record the next
                    # valid timestep regardless of dump settings
                    accept_next = True
            except IndexError as e:
                if line.startswith("Last command:"):
                    break
                else:
                    # Do not expect an IndexError otherwise, as expected EW lines will
                    # have sufficient length and so will cause a ValueError instead,
                    # so re-raise with more info if this ever happens.
                    raise IndexError(
                        f"Cannot access entries {step_index} or {header_index} "
                        f"from line number {header_line_number + 1 + i}: '{line}'"
                    ) from e

            i += 1
    output_list.shape = ((int(len(output_list) / len(output_headers)), len(output_headers)))

    # Set `extrapolation_free_lines` and `extrapolation_free_timesteps` if no extrapolation
    if not extrapolation:
        extrapolation_free_lines = -1
        extrapolation_free_timesteps = timesteps_list[-1]

    timesteps = np.unique(timesteps_list)
    dataset = {}

    for header_index, output_header in enumerate(output_headers):
        dataset[output_header] = output_list[:, header_index]

    if len(timesteps) == 0:
        # `timesteps` should have length here, as crashing between printing the headers
        # and the 0th timestep is unlikely. However, still raise an error for cases that
        # may have been missed.
        raise ValueError("No timesteps completed for {}".format(log_lammps_file))

    return (
        timesteps,
        extrapolation_free_lines,
        extrapolation_free_timesteps,
        dataset,
    )
