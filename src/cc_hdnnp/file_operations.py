"""
Utility functions for reading/writing to file, independent from `Data` or `ActiveLearning`
workflows.
"""

import re
from typing import Dict


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
