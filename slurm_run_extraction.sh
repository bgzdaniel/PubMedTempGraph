#!/bin/bash

#SBATCH -p dev
#SBATCH --exclude=graphcore,octane[001-008]
#SBATCH --output=data_extraction/slurm_extraction_output.txt

year=$1
srun -u python -m data_extraction.data_extraction --year ${year}