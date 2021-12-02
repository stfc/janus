#!/bin/bash
#SBATCH --job-name=active_learning_lammps
#SBATCH -C scarf18
#SBATCH -N 1-1
#SBATCH --ntasks-per-node=24 
#SBATCH -t 12:00:00
#SBATCH -o %j.out
#SBATCH --array=1-1000
#SBATCH --account=scddevel  

# TODO remove hardcoding
nn_exe='/home/vol00/scarf860/cc_placement/n2p2/bin'

mkdir ../active_learning/mode2
mkdir ../active_learning/mode2/HDNNP_1 ../active_learning/mode2/HDNNP_2

# TODO remove hardcoding
cp ../n2p2_1/input.nn ../active_learning/mode2/HDNNP_1/input.nn
cp ../n2p2_2/input.nn ../active_learning/mode2/HDNNP_2/input.nn

sed -i s/'.*test_fraction.*'/'test_fraction 0.0'/g mode2/HDNNP_*/input.nn
sed -i s/'epochs.*'/'epochs 0'/g mode2/HDNNP_*/input.nn
sed -i s/'.*use_old_weights_short'/'use_old_weights_short'/g mode2/HDNNP_*/input.nn
sed -i s/"#CAUTION: don't forget use_short_forces below (if you want to generate the training files for the forces)"/"#CAUTION: don't forget use-short-forces below (if you want to generate the training files for the forces)"/g mode2/HDNNP_*/input.nn
sed -i s/'.*use_short_forces'/'use_short_forces'/g mode2/HDNNP_*/input.nn
# Try leaving this in to improve normed performance?
# sed -i s/'.*atom_energy'/'#atom_energy'/g mode2/HDNNP_*/input.nn
sed -i s/'.*write_trainpoints'/'write_trainpoints'/g mode2/HDNNP_*/input.nn
sed -i s/'.*write_trainforces'/'write_trainforces'/g mode2/HDNNP_*/input.nn
sed -i s/'.*precondition_weights'/'#precondition_weights'/g mode2/HDNNP_*/input.nn
sed -i s/'.*nguyen_widrow_weights_short'/'#nguyen_widrow_weights_short'/g mode2/HDNNP_*/input.nn

cd ../active_learning/mode2/HDNNP_1
  ln -s ../../input.data-new input.data
  mpirun -np ${SLURM_NTASKS} ${nn_exe}/nnp-scaling 500 > mode_1.out &
cd ../../mode2/HDNNP_2
  ln -s ../../input.data-new input.data
  mpirun -np ${SLURM_NTASKS} ${nn_exe}/nnp-scaling 500 > mode_1.out &
cd ../..
wait

cp ../n2p2_1/scaling.data mode2/HDNNP_1
cp ../n2p2_1/weights.*.data mode2/HDNNP_1
cp ../n2p2_2/scaling.data mode2/HDNNP_2
cp ../n2p2_2/weights.*.data mode2/HDNNP_2

cd mode2/HDNNP_1
  mpirun -np ${SLURM_NTASKS} ${nn_exe}/nnp-train > mode_2.out &
cd ../../mode2/HDNNP_2
  mpirun -np ${SLURM_NTASKS} ${nn_exe}/nnp-train > mode_2.out &
cd ../..
wait

rm mode2/HDNNP_*/000000.short.*.out mode2/HDNNP_*/debug.out mode2/HDNNP_*/function.data mode2/HDNNP_*/testforces.000000.out mode2/HDNNP_*/testforces.data mode2/HDNNP_*/testing.data mode2/HDNNP_*/testpoints.000000.out mode2/HDNNP_*/teststruct.data mode2/HDNNP_*/trainforces.data mode2/HDNNP_*/trainstruct.data

####################################################################################################

####################################################################################################
