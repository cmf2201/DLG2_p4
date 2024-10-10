#!/bin/bash

#SBATCH --mail-user=username@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J window_seg
#SBATCH --output=out%j.out
#SBATCH --error=err%j.err

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -p academic
#SBATCH -t 23:00:00

module load miniconda3

source activate
conda activate patching2

module load cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 MainCondition.py