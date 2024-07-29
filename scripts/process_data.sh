#!/bin/sh
eval "$(conda shell.bash hook)"

conda activate olf

python3 process/extract_unique_cids.py
python3 process/process_percept.py
python3 process/process_descriptors.py