#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --mem=80Gb
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  # Request one GPU

#SBATCH --job-name=generate_data
#SBATCH --output slurm/generate_data.out

srun -n 1 python data/arc/prepare.py