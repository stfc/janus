#!/usr/bin/env bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24 
#SBATCH -C scarf18 
#SBATCH --job-name="nnp-train" 
#SBATCH -t 12:00:00 
#SBATCH --exclusive 
#SBATCH --account=scddevel  
#SBATCH --reservation=scddevel 
#SBATCH -o %j.out # STDOUT
#SBATCH -e %j.err # STDERR


module purge 

module use /work3/cse/dlp/eb-ml/modules/all
module load foss/2020a

cd ../n2p2

mpirun -n 24 /home/vol00/scarf860/cc_placement/n2p2/bin/nnp-norm
mpirun -n 24 /home/vol00/scarf860/cc_placement/n2p2/bin/nnp-scaling 500
mpirun -n 24 /home/vol00/scarf860/cc_placement/n2p2/bin/nnp-train
