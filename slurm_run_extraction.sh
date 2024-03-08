#!/bin/bash

#SBATCH -p dev
#SBATCH --exclude=graphcore,octane[001-008]
#SBATCH --output=data_extraction/slurm_extraction_output.txt

srun -u python -m data_extraction.data_extraction