# Olfboost
Learning Olfactory Mixture Similarity: D2Smell team

Project page: [Link](https://www.synapse.org/Synapse:syn61941777/wiki/629245)

## Getting started
### Data Preprocess
```.bash
sh scripts/scripts/process_data.sh
```

### Data Augmentation
Please follow the instructions on this [page](https://github.com/Satarifard/CWYK-Olfboost/tree/main/augmentation).

### Generate the Mixture Features
Please follow the instructions on this [page](https://github.com/Satarifard/CWYK-Olfboost/tree/main/odor-pair-model).

### Train and Inference of Precept Regressor
```.bash
python train/percept_regressor.py
```

### Train and Inference of Mixture Regressor
```.bash
python train/mixture_regressor.py
```

### Ensemble Models
```.bash
sh scripts/generate_batch.sh
sbatch scripts/ensemble_train.sh
sbatch scripts/ensemble.sh
```
