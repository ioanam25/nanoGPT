#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --mem=80Gb
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1  # Request one GPU

#SBATCH --job-name=sample
#SBATCH --output slurm/sample.out

srun -n 1 python sample.py