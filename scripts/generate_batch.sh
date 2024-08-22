#!/usr/bin/env bash

output_dir="scripts/batch"  

mkdir -p $output_dir

ids=(42 3407 5067 1111 2222 3333 6666 7777 8888 9999)
for id in "${ids[@]}"
do
  filename="${output_dir}/${id}.sh"
  
  cat <<EOL > $filename
#!/usr/bin/env bash
#SBATCH -A berzelius-2024-123
#SBATCH -t 0-1:0:0
#SBATCH --gres gpu:1
#SBATCH --mail-type "BEGIN,END,FAIL"
#SBATCH --mail-user "yinw@kth.se"
#SBATCH --output logs/${id}.log
#SBATCH --error logs/${id}.log

nvidia-smi
module load Anaconda/2021.05-nsc1
conda activate olf

python3 train/mixture_regressor_ensemble.py --seed ${id}
EOL

  chmod +x $filename
done