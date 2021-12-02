#!/usr/bin/env bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24 
#SBATCH -C scarf18 
#SBATCH --job-name="cresol_{file_id}" 
#SBATCH -t 12:00:00 
#SBATCH --exclusive 
#SBATCH --account=scddevel  
#SBATCH --reservation=scddevel 
#SBATCH -o %j.out # STDOUT
#SBATCH -e %j.err # STDERR


module purge 

module use /work3/cse/dlp/eb-ml/modules/all
module load CP2K/8.1-foss-2020b
srun cp2k.popt ../cp2k_input/cresol_{file_id}.inp &> ../cp2k_output/cresol_{file_id}.log
mv cresol_{file_id}-forces-1_0.xyz ../cp2k_output/cresol_{file_id}-forces-1_0.xyz
