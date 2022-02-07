###############################################################################
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

###############################################################################
# INTEGRATOR
###############################################################################
velocity     all create {temp} 2020 mom yes rot yes dist gaussian
velocity all zero angular
velocity all zero linear
fix x all nve
{dump_commands}
run {n_steps}
unfix x

###############################################################################
# INTEGRATOR
###############################################################################
velocity     all create {temp} 2020 mom yes rot yes dist gaussian
velocity all zero angular
velocity all zero linear
fix y all nvt temp {temp} {temp} {tdamp}
{dump_commands}
run {n_steps}
unfix y

###############################################################################
# INTEGRATOR
###############################################################################
velocity     all create {temp} 2020 mom yes rot yes dist gaussian
velocity all zero angular
velocity all zero linear
fix z all npt temp {temp} {temp} {tdamp} iso {pressure} {pressure} {pdamp}  tchain 3
{dump_commands}
run {n_steps}
unfix z
