#!/usr/bin/env bash
#SBATCH -A berzelius-2024-123
#SBATCH -t 0-1:0:0
#SBATCH --gres gpu:1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "yinw@kth.se"
#SBATCH --output logs/8888.log
#SBATCH --error logs/8888.log

nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate olf

python3 train/mixture_regressor_ensemble.py --seed 8888
