"""
Unit tests for `file_operations.py`
"""
from os import remove

from cc_hdnnp.slurm_input import format_slurm_input


def test_template_file():
    """
    Test that if a template file is provided, it is used.
    """
    template_file = "tests/data/tests_output/template.txt"
    formatted_file = "tests/data/tests_output/formatted.txt"
    with open(template_file, "w") as f:
        f.write("the template text\n{main_commands}")

    format_slurm_input(
        template_file=template_file,
        formatted_file=formatted_file,
        commands=["command 1", "command 2"],
    )

    with open(formatted_file) as f:
        text = f.read()
    assert text == "the template text\ncommand 1\ncommand 2"

    remove(template_file)
    remove(formatted_file)


def test_optional_arguments():
    """
    Test that if optional SBATCH arguments without defaults
    (`array`, `account`, `reservation`, `exclusive`) are given, they are formatted correctly.
    """
    formatted_file = "tests/data/tests_output/formatted.txt"
    format_slurm_input(
        formatted_file=formatted_file,
        commands=["command 1", "command 2"],
        array="0-1",
        account="account_name",
        reservation="reservation_name",
        exclusive=True,
    )

    with open(formatted_file) as f:
        text = f.read()
    assert "\n#SBATCH --array=0-1\n" in text
    assert "\n#SBATCH --account=account_name\n" in text
    assert "\n#SBATCH --reservation=reservation_name\n" in text
    assert "\n#SBATCH --exclusive\n" in text

    remove(formatted_file)
