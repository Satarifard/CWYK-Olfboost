#!/usr/bin/env bash
#SBATCH -A berzelius-2024-123
#SBATCH -t 0-1:0:0
#SBATCH --gres gpu:1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "yinw@kth.se"
#SBATCH --output logs/ens.log
#SBATCH --error logs/ens.log

nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate olf

python3 train/ensemble.py