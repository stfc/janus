#!/usr/bin/env bash 

for i in {0..100..1}
do
   sbatch cp2k_batch_n_${i}_cutoff_600_relcutoff_60.bash
done