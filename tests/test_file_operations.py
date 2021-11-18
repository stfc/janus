"""
Unit tests for `file_operations.py`
"""
import pytest

from cc_hdnnp.file_operations import (
    format_template_file,
)


@pytest.mark.parametrize(
    "format_shell_variables, text_in, text_out",
    [
        (True, "{normal} ${shell}", "one two"),
        (False, "{normal} ${shell}", "one $two"),
    ],
)
def test_format_template_file(
    format_shell_variables: bool, text_in: str, text_out: str
):
    """
    Test that the file is formatted correctly depending on `format_shell_variables`.
    """
    template_file = "tests/data/tests_output/template.txt"
    formatted_file = "tests/data/tests_output/formatted.txt"
    with open(template_file, "w") as f:
        f.write(text_in)

    format_template_file(
        template_file=template_file,
        formatted_file=formatted_file,
        format_dict={"normal": "one", "shell": "two"},
        format_shell_variables=format_shell_variables,
    )

    with open(formatted_file) as f:
        text = f.read()
    assert text == text_out
