"""
Reads lammps.log files for information used in the workflow.
"""

from typing import Tuple

import numpy as np


def read_lammps_log(
    dump_lammpstrj: int, log_lammps_file: str
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """
    Reads a "log.lammps"and extracts information about if and at
    what timestep extrapolation of the network potential occurred.

    Parameters
    ----------
    dump_lammpstrj : int
        Integer which defines that only every nth frame of the simulation is returned if no
        extrapolation occurred.
    log_lammps_file : str
        The file path to the "log.lammps" file.

    Returns
    -------
    (np.ndarray, int, int, np.ndarray)
        First element is array of int corresponding to timesteps, second is the number of
        extrapolation free lines and the third is the timestep that corresponds to that
        line. The fourth is the temperate at each timestep of the simulation.
    """
    with open(log_lammps_file) as f:
        data = f.readlines()

    if len(data) == 0:
        raise ValueError("{} was empty".format(log_lammps_file))

    # Count the number of lines that precede the simulation so they can be skipped
    step_index = 0
    temp_index = 1
    epair_index = -1
    emol_index = -1
    etot_index = -1
    press_index = -1
    vol_index = -1
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
    temp_index = headers.index("Temp")
    if("E_pair" in headers):epair_index = headers.index("E_pair")
    if("E_mol" in headers):emol_index = headers.index("E_mol")
    if("TotEng" in headers):etot_index = headers.index("TotEng")
    if("Press" in headers):press_index = headers.index("Press")
    if("Volume" in headers):vol_index = headers.index("Volume")
    timesteps_list = []
    temp_list = []
    epair_list = []
    emol_list = []
    etot_list = []
    press_list = []
    vol_list = []
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
                temp_list.append(float(line.split()[temp_index]))
                if(epair_index>=0):epair_list.append(float(line.split()[epair_index]))
                if(emol_index>=0):emol_list.append(float(line.split()[emol_index]))
                if(etot_index>=0):etot_list.append(float(line.split()[etot_index]))
                if(press_index>=0):press_list.append(float(line.split()[press_index]))
                if(vol_index>=0):vol_list.append(float(line.split()[vol_index]))
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
                # Do not expect an IndexError here, as expected EW lines will have sufficient
                # length and so will cause a ValueError instead, so re-raise with more info if
                # this ever happens.
                raise IndexError(
                    f"Cannot access entries {step_index}, {temp_index} from line number "
                    f"{header_line_number + 1 + i}: '{line}'"
                ) from e

            i += 1

    # Set `extrapolation_free_lines` and `extrapolation_free_timesteps` if no extrapolation
    if not extrapolation:
        extrapolation_free_lines = -1
        extrapolation_free_timesteps = timesteps_list[-1]

    # temperatures = np.array(temp_list)
    timesteps = np.unique(timesteps_list)
    dataset = {}
    dataset["Temp"] = np.array(temp_list)
    if("E_pair" in headers):dataset["E_pair"] = np.array(epair_list)
    if("E_mol" in headers):dataset["E_mol"] = np.array(emol_list)
    if("TotEng" in headers):dataset["TotEng"] =np.array(etot_list)
    if("Press" in headers):dataset["Press"] = np.array(press_list)
    if("Volume" in headers):dataset["Volume"] = np.array(vol_list)


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
