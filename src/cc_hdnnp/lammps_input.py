"""
Utility functions for formatting a LAMMPS input script.
"""


from typing import Iterable


TEMPLATE = """###############################################################################
# MD simulation
###############################################################################

###############################################################################
# GENERAL SETUP
###############################################################################
units {units}
boundary p p p
atom_style {atom_style}
read_data {read_data}
{masses}
timestep {timestep}
thermo 1
###############################################################################
# NN
###############################################################################
pair_style nnp dir {nnp_dir} showew {showew} showewsum {showewsum} resetew {resetew} maxew {maxew} emap {emap} cflength {cflength} cfenergy {cfenergy}
pair_coeff * * {pair_coeff}
{integrator_commands}
""" 


INTEGRATOR_NVE = """###############################################################################
# INTEGRATOR
###############################################################################
velocity     all create {temp} {seed} mom yes rot yes dist gaussian
velocity all zero angular
velocity all zero linear
fix x all nve
{dump_commands}
run {n_steps}
unfix x
"""

INTEGRATOR_NVT = """###############################################################################
# INTEGRATOR
###############################################################################
velocity     all create {temp} {seed} mom yes rot yes dist gaussian
velocity all zero angular
velocity all zero linear
fix y all nvt temp {temp} {temp} {tdamp}
{dump_commands}
run {n_steps}
unfix y
"""

INTEGRATOR_NPT = """###############################################################################
# INTEGRATOR
###############################################################################
velocity     all create {temp} {seed} mom yes rot yes dist gaussian
velocity all zero angular
velocity all zero linear
fix z all npt temp {temp} {temp} {tdamp} {barostat} {pressure} {pressure} {pdamp} {npt_other}
{dump_commands}
run {n_steps}
unfix z
"""  # noqa: E501

DUMP_CUSTOM = """dump            cust all custom 1 {dump_file} element x y z vx vy vz fx fy fz
dump_modify     cust element {elements}
"""

DUMP_XYZ = """dump            xyzmov all xyz {xyz_sampling} {dump_file}
dump_modify     xyzmov element {elements}
"""

DUMP_VEL = """dump            velmov all custom  1 {dump_file} element x y z vx vy vz
dump_modify     velmov element {elements}
dump_modify     velmov delay {vel_steps}
dump_modify     velmov sort id
"""


def format_lammps_input(
    formatted_file: str,
    masses: str,
    emap: str,
    template_file: str = None,
    units: str = "metal",
    atom_style: str = "atomic",
    read_data: str = "lammps.data",
    timestep: str = "0.0005",
    nnp_dir: str = "nnp",
    showew: str = "no",
    showewsum: str = "10",
    resetew: str = "no",
    maxew: str = "100",
    cflength: str = "1.889726125836928",
    cfenergy: str = "0.03674932247495664",
    pair_coeff: str = "6.351",
    n_steps: Iterable[str] = (50000,),
    integrators: Iterable[str] = ("nve",),
    temps: Iterable[str] = ("300",),
    tdamp: str = "10",
    seed: int = 1,
    barostat: str = "iso",
    pressure: str = "0",
    pdamp: str = "50",
    npt_other: str = "tchain 3",
    dump_commands: str = None,
    elements: str = None,
    dump_file: str = "dump.lammpstrj",
    
):
    """
    Formats a LAMMPS input script with the provided arguments, either based on a provided
    `template_file` or using a default setup.

    Most arguments have default values, however arguments relating to the elements in the
    structure (`masses`, `emap`, `elements`) must be provided.

    Specific commands for how to dump information to file can be used by providing
    `dump_commands`, otherwise `elements` must be provided and information will be
    dumped to `dump_file` by default.

    Parameters
    ----------
    formatted_file: str
        The complete filepath to write the formatted input script to.
    masses: str
        A string representing the masses of the elements. It should be of the format:
        "mass 1 1.00794\nmass 2 15.9994\n"
        where elements are sorted alphabetically, and separated by newlines.
    emap: str
        A string representing the mapping of indices to elements. It should be of the format:
        "1:H,2:O"
        where elements are sorted alphabetically.
    template_file: str = None
        If provided, this file will be used as a template to format with the provided arguments
        rather than the default template. Optional, default is None.
    units: str = "metal"
        The LAMMPS units string. Optional, default is "metal".
    atom_style: str = "atomic"
        The LAMMPS style of atoms. Optional, default is "atomic".
    read_data: str = "lammps.data"
        Filepath to the LAMMPS data file. Optional, default is "lammps.data".
    timestep: str = "0.0005"
        Time taken for each step of the simulation. Units will depend on `units`.
        Optional, default is "0.0005".
    nnp_dir: str = "nnp"
        Filepath to the directory containing the neural network files.
        Optional, default is "nnp".
    showew: str = "no"
        Whether to show extrapolation warnings, with details on the magnitude of extrapolation,
        every time the occur. Optional, default is "no".
    showewsum: str = "10"
        How often to show a summary of extrapolation warnings, in terms of number of timesteps.
        Optional, default is "10".
    resetew: str = "no"
        Whether reset the running count of extrapolation warnings at the start of each
        timestep ("yes"), or to accumulate warnings along the whole trajectory ("no").
        Optional, default is "no".
    maxew: str = "100"
        The maximum number of extrapolation warnings to allow before stopping the simulation.
        Optional, default is "100".
    cflength: str = "1.889726125836928"
        Factor to convert from neural network units to LAMMPS units. 1 LAMMPS length unit
        should equal `cflength` network units. Optional, default is "1.889726125836928",
        the factor to convert from a network using Bohr to LAMMPS using Ang.
    cfenergy: str = "0.03674932247495664"
        Factor to convert from neural network units to LAMMPS units. 1 LAMMPS energy unit
        should equal `cfenergy` network units. Optional, default is "0.03674932247495664",
        the factor to convert from a network using Ha to LAMMPS using eV.
    pair_coeff: str = "6.351"
        Radial cutoff, in LAMMPS length units. Optional, default is "6.351".
    n_steps: Iterable[str] = (50000,)
        Number of steps to take in the simulation. If multiple arguments are provided,
        each will be used for a simulation in turn with the relevant entry in
        `integrators` and `temps`. Optional, default is ("50000",).
    integrator: Iterable[str] = ("nve",)
        Ensemble to use, should be one of "nve", "nvt" or "npt". If multiple arguments
        are provided, each will be used for a simulation in turn with the relevant entry in
        `temps` and `n_steps`. Optional, default is ("nve",).
    temps: Iterable[str] = ("300",)
        Temperature to use for the simulation. If multiple arguments are provided,
        each will be used for a simulation in turn with the relevant entry in
        `integrators` and `n_steps`. Optional, default is ("300",).
    tdamp: str = "10"
        The timespan of temperature relaxations, in LAMMPS time units.
        Optional, default is "10".
    seed: int = 1
        The seed to use for the velocity creation. Optional, default is 1.
    barostat: str = "iso"
        The barostat option is only used when `integrator=="npt"` and should be one of
        "iso", "aniso", "tri". Optional, default is "iso".
    pressure: str = "0"
        Pressure to use for the simulation. Not used if `integrator` is "nve" or "npt".
        Optional, default is "0".
    pdamp: str = "50"
        The timespan of pressure relaxations, in LAMMPS time units.
        Optional, default is "50".
    npt_other: str = "tchain 3"
        Additional keyword commands to include when using the "npt" ensemble.
        Only used `integrator=="npt"`. For possible values refer to LAMMPS documentation.
        Optional, default is "tchain 3".
    dump_commands: str = None
        Commands for dumping to file. Optional, default is None in which case the template
        format for dumping will be used, and elements must be provided in order to format it.
    elements: str = None
        The chemical symbols of the elements in the structure, sorted alphabetically and
        separated by spaces. Optional, default is None, but if `dump_commands` is None
        `elements` must not be None.
    dump_file: str = "custom.dump"
        Filepath to dump to. Only used if `dump_commands` is None.
        Optional, default is "custom.dump".
    """
    if template_file is not None:
        with open(template_file) as f:
            template_text = f.read()
    else:
        template_text = TEMPLATE
    
    if isinstance(n_steps, int):
        n_steps = [n_steps]
    if isinstance(integrators, str):
        integrators = [integrators]
    if isinstance(temps, str):
        temps = [temps]
    dump = ""
    integrator_commands = ""
    for i, integrator in enumerate(integrators):
        temp = temps[i]
        n_step_i = n_steps[i]
        if i == len(integrators) - 1:
            if dump_commands is None and elements is None:
                raise ValueError(
                    "Either `dump_commands` or `elements` must not be None, but both were."
                )
            elif dump_commands is None and elements is not None:
                # dump_commands = DUMP_CUSTOM.format(dump_file=dump_file, elements=elements)
                dump_commands = DUMP_XYZ.format(
                    dump_file=f"{integrator}_t{temp}_xyz_" + dump_file,
                    elements=elements,
                    xyz_sampling=max(n_step_i // 10000, 1),
                )
                dump_commands += DUMP_VEL.format(
                    dump_file=f"{integrator}_t{temp}_vel_" + dump_file,
                    elements=elements,
                    vel_steps=max(n_step_i - 10000, 0),
                )
            dump = dump_commands
        if integrator == "nve":
            integrator_commands += INTEGRATOR_NVE.format(
                seed=seed, temp=temp, tdamp=tdamp, dump_commands=dump, n_steps=n_step_i
            )
        elif integrator == "nvt":
            integrator_commands += INTEGRATOR_NVT.format(
                seed=seed, temp=temp, tdamp=tdamp, dump_commands=dump, n_steps=n_step_i
            )
        elif integrator == "npt":
            if barostat not in ("iso", "aniso", "tri"):
                raise ValueError(f"Barostat option '{barostat}' is not implemented.")
            integrator_commands += INTEGRATOR_NPT.format(
                seed=seed,
                temp=temp,
                tdamp=tdamp,
                barostat=barostat,
                pressure=pressure,
                pdamp=pdamp,
                npt_other=npt_other,
                dump_commands=dump,
                n_steps=n_step_i,
            )
        else:
            raise ValueError(
                f"`integrator` must be one of 'nve', 'nvt' or 'npt', but was '{integrator}'"
            )
    output_text = template_text.format(**locals())

    with open(formatted_file, "w") as f:
        f.write(output_text)
