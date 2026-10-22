#!/bin/bash
# Only if your server uses Slurm:  sbatch job.sbatch
#SBATCH --job-name=en4554_a2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
mkdir -p logs
# source ~/venvs/a2/bin/activate     # <- activate your environment here
./run_all.sh
