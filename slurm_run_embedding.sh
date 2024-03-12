#!/bin/bash

#SBATCH --exclude=graphcore,ceg-victoria,octane[001-008]
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH --output=abstract2chroma/slurm_embedding_output_%j.txt

year=$1
srun -u python -m abstract2chroma.abstract2vec --year ${year}