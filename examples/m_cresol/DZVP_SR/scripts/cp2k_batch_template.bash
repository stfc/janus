#!/usr/bin/env bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24 
#SBATCH -C scarf18 
#SBATCH --job-name="DZVP_SR_{0}" 
#SBATCH -t 12:00:00 
#SBATCH --exclusive 
##SBATCH --account=scddevel  
##SBATCH --reservation=scddevel 
#SBATCH -o out/DZVP_SR_batch_%j.out # STDOUT
#SBATCH -e err/DZVP_SR_batch_%j.err # STDERR


module purge 

module load CP2K/6.1-foss-2019a 
srun cp2k.popt ../cp2k_input/cresol_DZVP_SR_{0}.inp &> ../cp2k_output/n1_cresol_DZVP_SR_{0}.log