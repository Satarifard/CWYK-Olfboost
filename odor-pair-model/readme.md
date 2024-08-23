## Steps to Generate the 33-Label Odor-Pair Model and Mixture Prediction File

1. **Download the Odor-Pair Script:**  
   Clone or download the odor-pair script from the following repository:  
   [https://github.com/odor-pair/odor-pair](https://github.com/odor-pair/odor-pair)

2. **Train the 33 Odor-Pair Models:**  
   Train the models using the provided script. Pre-trained models (`model_33.pt`) and the corresponding labels list (`labels.csv`) are available in the current directory. The AUROC values for each model are also provided.

3. **Test Odor-Pair Model Performance:**  
   To evaluate the performance of the model (AUROC), use the `ModelPerformance.ipynb` notebook included in this repository.

4. **Generate 33-Label Odor-Pair Predictions for Mixtures:**  
   For generating 33-label odor-pair predictions for any mixture, use the `33label_mixture_prediction.ipynb` notebook.

For more details, please refer to the [original paper](https://arxiv.org/html/2312.16124v1).
