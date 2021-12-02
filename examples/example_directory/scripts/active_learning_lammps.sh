#!/bin/bash
#SBATCH --job-name=active_learning_lammps
#SBATCH -C scarf18
#SBATCH -N 1-1
#SBATCH --ntasks-per-node=24 
#SBATCH -t 12:00:00
#SBATCH -o %j.out
#SBATCH --array=1-1000
#SBATCH --account=scddevel  

path=$(sed -n ${SLURM_ARRAY_TASK_ID}p ${SLURM_SUBMIT_DIR}/../active_learning/joblist_mode1.dat)

ulimit -s unlimited
module purge 
module use /work3/cse/dlp/eb-ml/modules/all
module load foss/2020a

dir=$(date '+%Y%m%d_%H%M%S_%N')
mkdir -p /scratch/${whoami}/${dir}
rsync -a ${SLURM_SUBMIT_DIR}/../active_learning/mode1/${path}/* /scratch/scarf860/${dir}/
cd /scratch/${whoami}/${dir}

# TODO remove hardcoding
mpirun -np ${SLURM_NTASKS} /home/vol00/scarf860/cc_placement/lammps/build/lmp_mpi -in input.lammps -screen none

# TODO move cleanup to a seperate sh script?
# ${SLURM_SUBMIT_DIR}/RuNNerActiveLearn.py 1a

rm -r /scratch/${whoami}/${dir}/RuNNer
rsync -a /scratch/${whoami}/${dir}/* ${SLURM_SUBMIT_DIR}/../active_learning/mode1/${path}/
cd ${SLURM_SUBMIT_DIR}/../active_learning/mode1/${path}
if [ "${dir}" != "" ]
then
  rm -r /scratch/${whoami}/${dir}
fi