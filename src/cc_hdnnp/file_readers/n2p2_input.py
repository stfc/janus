"""
Reads n2p2 input.nn files for information used in the workflow.
"""

from typing import Dict, List


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
                # Normalisation settings may not be present,
                # in which case return strings of 0 and 1
                if setting == "mean_energy":
                    returned_settings[setting] = "0"
                elif setting in ("conv_energy", "conv_length"):
                    returned_settings[setting] = "1"
                else:
                    print(
                        "Could not find setting {0} in {1}".format(
                            setting, settings_file
                        )
                    )

    return returned_settings
