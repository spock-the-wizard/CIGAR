#!/bin/bash

#SBATCH -J BERT_generate # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p TITANRTX # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --nodes=1
#SBATCH  --ntasks=4
#SBATCH  --cpus-per-task=8

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

python attribute_vector.py --mode bert
python assign_captions.py --vector_type bert

date