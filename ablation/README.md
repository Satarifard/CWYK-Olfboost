# Ablation Study

Feature ablation for the CWYK-Olfboost mixture similarity model. Evaluates the contribution of two feature groups by training and predicting with each in isolation:

- **Non-Semantic (`class1`):** Binary mixture composition vectors only — one-hot presence/absence of each molecule (~370 features).
- **Semantic (`class23`):** Predicted perceptual descriptors (squared differences between mixtures) + expanded molecular descriptor features (per-component perceptual profiles) (~7,500 features).

Everything else (Optuna hyperparameter tuning, 10-fold CV, data augmentation, ensemble averaging) is identical to the original pipeline in `train/`.

## Reproduce

From the repo root:

```bash
cd /Ablation/CWYK-Olfboost-main
```

### 1. Train (10 seeds x 2 conditions)

```bash
bash ablation/run_training.sh
```

This trains 20 models total (~15 min per seed for class23, ~3 min for class1 on CPU).
Models are saved to `output/ablation_class1/` and `output/ablation_class23/`.

### 2. Ensemble predictions

```bash
bash ablation/run_ensemble.sh
```

Outputs saved to `final_results/`:
- `Leaderboard_set_Submission_form_ensemble_class1.csv`
- `Test_set_Submission_form_ensemble_class1.csv`
- `Leaderboard_set_Submission_form_ensemble_class23.csv`
- `Test_set_Submission_form_ensemble_class23.csv`
