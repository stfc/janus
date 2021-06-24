#!/usr/bin/env bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24 
#SBATCH -C scarf18 
#SBATCH --job-name="TZVP_0" 
#SBATCH -t 12:00:00 
#SBATCH --exclusive 
##SBATCH --account=scddevel  
##SBATCH --reservation=scddevel 
#SBATCH -o out/TZVP_batch_%j.out # STDOUT
#SBATCH -e err/TZVP_batch_%j.err # STDERR


module purge 

module load CP2K/6.1-foss-2019a 
srun cp2k.popt ../cp2k_input/cresol_TZVP_0.inp &> ../cp2k_output/n1_cresol_TZVP_0.log