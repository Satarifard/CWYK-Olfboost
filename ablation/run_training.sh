#!/bin/bash
for seed in 42 1111 2222 3333 3407 5067 6666 7777 8888 9999; do
    python3 ablation/mixture_regressor_ensemble.py --seed $seed --ablation class1
    python3 ablation/mixture_regressor_ensemble.py --seed $seed --ablation class23
done
