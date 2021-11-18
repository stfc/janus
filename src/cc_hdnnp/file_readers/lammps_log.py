"""
Reads lammps.log files for information used in the workflow.
"""

from typing import Tuple

import numpy as np


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
    step_index = 0
    header_line_number = 0
    n_lines = len(data)
    loop_time_line_number = n_lines
    while header_line_number < n_lines and "Step" not in data[header_line_number]:
        header_line_number += 1

    if header_line_number == n_lines:
        raise ValueError(
            f"{log_lammps_file} does not contain simulation headers, "
            "indicating it crashed before the 0th timestep."
        )
    headers = data[header_line_number].split()
    step_index = headers.index("Step")

    # Starting at `counter`, check for extrapolation warnings
    extrapolation = False
    i = header_line_number + 1
    while i < n_lines:
        if data[i].startswith("Loop time of "):
            loop_time_line_number = i
            break
        elif (
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
            extrapolation = True
            break
        else:
            i += 1

    # From the extrapolation warning (or end of simulation) look backwards to see how many steps
    # occured
    i -= 1
    # TODO REMOVE this once sure we don't need it
    # while i > header_line_number and (
    #     data[i].startswith("### NNP EXTRAPOLATION WARNING ###")
    #     or data[i].startswith("### NNP EXTRAPOLATION SUMMARY ###")
    #     or data[i].startswith("### NNP EW SUMMARY ###")
    #     or data[i].startswith("WARNING:")
    # ):
    #     i -= 1
    if extrapolation:
        extrapolation_free_lines = i
        if i > header_line_number:
            extrapolation_free_timesteps = int(data[i].split()[step_index])
        else:
            # Should not be able to reach this, as 0th timestep is written immediately
            # after the headers, so assign -1
            extrapolation_free_timesteps = -1
    else:
        extrapolation_free_lines = -1
        extrapolation_free_timesteps = int(data[i].split()[step_index])

    # Subsample using `dump_lammpstrj`
    timesteps_list = []
    accept_next = False
    for line in data[header_line_number + 1 : loop_time_line_number]:
        try:
            timestep = int(line.split()[step_index])
            if accept_next or timestep % dump_lammpstrj == 0:
                timesteps_list.append(timestep)
                accept_next = False
        except ValueError:
            # If we cannot cast to int due to an EW, then record the next valid timestep
            # regardless of dump settings
            if (
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
                accept_next = True
            pass
        except IndexError as e:
            # Do not expect an IndexError here, as expected EW lines will have sufficient
            # length and so will cause a ValueError instead, so re-raise with more info if
            # this ever happens.
            raise IndexError(
                f"Cannot access entry {step_index} from the line '{line}'"
            ) from e

    timesteps = np.unique(timesteps_list)
    if len(timesteps) == 0:
        # `timesteps` should have length here, as crashing between printing the headers
        # and the 0th timestep is unlikely. However, still raise an error for cases that
        # may have been missed.
        raise ValueError("No timesteps completed for {}".format(log_lammps_file))

    return timesteps, extrapolation_free_lines, extrapolation_free_timesteps
