"""
Utility functions for formatting a SLURM batch script.
"""


from typing import Iterable


TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -C {constraint}
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH -t {time}
#SBATCH -o {out}
{array_command}
{account_command}
{reservation_command}
{exclusive_command}
{export_command}

module purge

{main_commands}
"""


def format_slurm_input(
    formatted_file: str,
    commands: Iterable[str],
    template_file: str = None,
    job_name: str = "cc_hdnnp_job",
    constraint: str = "intel",
    nodes: str = "1",
    ntasks_per_node: str = "24",
    time: str = "12:00:00",
    out: str = "%j.out",
    array: str = None,
    account: str = None,
    reservation: str = None,
    exclusive: bool = False,
    export: str = None,
):
    """
    Formats a SLURM batch script with the provided arguments, either based on a provided
    `template_file` or using a default setup.

    Most arguments have default values, however at least some `commands` must be provided.

    For more details on the SBATCH optional arguments, and their format, see the SLURM
    documentation.

    Parameters
    ----------
    formatted_file: str
        The complete filepath to write the formatted batch script to.
    commands: Iterable[str]
        The main commands to run. Each entry in the `Iterable` will be written to file as a
        different line. This should also include commands to load the needed modules.
    template_file: str = None
        If provided, this file will be used as a template to format with the provided arguments
        rather than the default template. Optional, default is None.
    job_name: str = "cc_hdnnp_job"
        The name to give the SLURM job. Optional, default "cc_hdnnp_job".
    constraint: str = "intel"
        The constraint to apply to node selection. Optional, default is "intel".
    nodes: str = "1"
        The number of nodes to request. Optional, default is "1".
    ntasks_per_node: str = "24"
        The number of tasks per node. Optional, default is "24".
    time: str = "12:00:00"
        The maximum time to run the job for. Optional, default is "12:00:00".
    out: str = "%j.out"
        The path to write the output of the job to. Optional, default is "%j.out".
    array: str = None
        If not None, then is used as the argument for the `--array` command, allowing
        multiple jobs to be submitted from the same script. Optional, default is None.
    account: str = None
        If not None, then is used as the argument for the `--account` command.
        Optional, default is None.
    reservation: str = None
        If not None, then is used as the argument for the `--reservation` command.
        Optional, default is None.
    exclusive: bool = False
        If not True, then the argument `--exclusive` is passed.
        Optional, default is False.
    export: str = None
        Environment variables exported.
        Optional, default is None.
    """
    if template_file is not None:
        with open(template_file) as f:
            template_text = f.read()
    else:
        template_text = TEMPLATE

    if array is not None:
        array_command = f"#SBATCH --array={array}"
    else:
        array_command = ""
    if account is not None:
        account_command = f"#SBATCH --account={account}"
    else:
        account_command = ""
    if reservation is not None:
        reservation_command = f"#SBATCH --reservation={reservation}"
    else:
        reservation_command = ""
    if exclusive:
        exclusive_command = "#SBATCH --exclusive"
    else:
        exclusive_command = ""
    if export:
        export_command = f"#SBATCH --export={export}"
    else:
        export_command = ""

    main_commands = "\n".join(commands)

    output_text = template_text.format(**locals())

    with open(formatted_file, "w") as f:
        f.write(output_text)
