# Olfboost
Learning Olfactory Mixture Similarity: CWYK team

Project page: [Link](https://www.synapse.org/Synapse:syn61941777/wiki/629245)

## Getting started
### Data Preprocess
```.bash
sh scripts/scripts/process_data.sh
```

### Train and Inference of Precept Regressor
```.bash
python train/percept_regressor.py
```

### Train and Inference of Mixture Regressor
```.bash
python train/mixture_regressor.py
```

### Pretrained Models
The pretrained models are located at ./pretrained. 
