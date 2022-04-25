#!/bin/bash

#SBATCH -J Nhot_0.5_train # job name
#SBATCH -o /home/deokhk/coursework/CIGAR/image_retrieval/sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p A5000 # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --nodes=1
#SBATCH  --ntasks=4
#SBATCH  --gres=gpu:1

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date
sh deepfashion_onehot_0.5.sh
date