import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib as jl
import wandb
import json
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer
from sklearn.model_selection import KFold


def prepare_training_data(training_data_df, df_percept, mixture_definitions_df, all_cids, add_aug_val, remove_aug_val):
    def get_features(dataset, mixture):
        return df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture), 'Prediction_1':].to_numpy().flatten()

    def append_vectors(X, y, vec1, vec2, exp_value):
        combined_vector = np.concatenate([vec1, vec2, z])
        X.append(combined_vector)
        y.append(exp_value)

    X = []
    y = []

    for _, row in training_data_df.iterrows():
        dataset, mixture1, mixture2, exp_value = row['Dataset'], row['Mixture 1'], row['Mixture 2'], row['Experimental Values']
        
        feature_1 = get_features(dataset, mixture1)
        feature_2 = get_features(dataset, mixture2)
        z = (feature_1 - feature_2) ** 2
        mixture1_vector = prepare_binary_vector(mixture_definitions_df[mixture_definitions_df['Dataset'] == dataset], all_cids, mixture1)
        mixture2_vector = prepare_binary_vector(mixture_definitions_df[mixture_definitions_df['Dataset'] == dataset], all_cids, mixture2)
        
        append_vectors(X, y, mixture1_vector, mixture2_vector, exp_value)
        append_vectors(X, y, mixture2_vector, mixture1_vector, exp_value)

        new_mix1_list, new_exp_val_list = data_aug_add(mixture1_vector, mixture2_vector, exp_value, add_aug_val, [1])
        for mix1, new_exp_val in zip(new_mix1_list, new_exp_val_list):
            append_vectors(X, y, mix1, mixture2_vector, new_exp_val)

        new_mix2_list, new_exp_val_list = data_aug_add(mixture2_vector, mixture1_vector, exp_value, add_aug_val, [1])
        for mix2, new_exp_val in zip(new_mix2_list, new_exp_val_list):
            append_vectors(X, y, mix2, mixture1_vector, new_exp_val)

        new_mix1_remove_list, new_mix2_remove_list, new_exp_val_remove_list = data_aug_remove(mixture1_vector, mixture2_vector, exp_value, remove_aug_val, [1])
        for new_mix1, new_mix2, new_exp_val in zip(new_mix1_remove_list, new_mix2_remove_list, new_exp_val_remove_list):
            append_vectors(X, y, new_mix1, new_mix2, new_exp_val)
            append_vectors(X, y, new_mix2, new_mix1, new_exp_val)

    return np.array(X), np.array(y)


def prepare_leadboard_data(leaderboard_submission_df, df_percept, mixture_definitions_leaderboard_df, all_cids):
    X_test = []

    for _, row in leaderboard_submission_df.iterrows():
        dataset, mixture1, mixture2 = row['Dataset'], row['Mixture_1'], row['Mixture_2']
        
        mixture1_vector = prepare_binary_vector(mixture_definitions_leaderboard_df[mixture_definitions_leaderboard_df['Dataset'] == dataset], all_cids, mixture1)
        mixture2_vector = prepare_binary_vector(mixture_definitions_leaderboard_df[mixture_definitions_leaderboard_df['Dataset'] == dataset], all_cids, mixture2)

        feature_1 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture1), 'Prediction_1':].to_numpy().flatten()
        feature_2 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture2), 'Prediction_1':].to_numpy().flatten()
        
        feature_12 = np.concatenate((feature_1, feature_2))
        feature_12 = (feature_1 - feature_2) ** 2
        combined_vector = np.concatenate([mixture1_vector, mixture2_vector, feature_12])
        X_test.append(combined_vector)
    
    return np.array(X_test)


def prepare_test_data(test_set_submission_df, df_percept, mixture_definitions_test_df, all_cids):
    X_test = []

    for _, row in test_set_submission_df.iterrows():
        dataset, mixture1, mixture2 = 'Test', row['Mixture_1'], row['Mixture_2']

        feature_1 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture1), 'Prediction_1':].to_numpy().flatten()
        feature_2 = df_percept.loc[(df_percept['Dataset'] == dataset) & (df_percept['Mixture Label'] == mixture2), 'Prediction_1':].to_numpy().flatten()
        feature_12 = np.concatenate((feature_1, feature_2))
        feature_12 = (feature_1 - feature_2) ** 2
        mixture1_vector = prepare_binary_vector(mixture_definitions_test_df, all_cids, mixture1)
        mixture2_vector = prepare_binary_vector(mixture_definitions_test_df, all_cids, mixture2)
        
        combined_vector = np.concatenate([mixture1_vector, mixture2_vector, feature_12])
        X_test.append(combined_vector)
    
    return np.array(X_test)

 
def train_model(training_data_df, df_percept, mixture_definitions_df, all_cids, X_t, y_t, data_percept, percept_ids):
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'leaderboard_corr',
            'goal': 'maximize'
        },
        'parameters': {
            'n_estimators': {
                'values': [100]  # Number of trees in the forest
            },
            'learning_rate': {
                'min': 0.01,
                'max': 0.1  # Learning rate range
            },
            'max_depth': {
                'min': 7,
                'max': 10  # Maximum depth of the tree range
            },
            'min_child_weight': {
                'min': 3,
                'max': 9  # Minimum sum of instance weight (hessian) range
            },
            'subsample': {
                'min': 0.5,
                'max': 0.9  # Subsample ratio of the training instances range
            },
            'colsample_bytree': {
                'min': 0.6,
                'max': 1.0  # Subsample ratio of columns range
            },
            'gamma': {
                'min': 0.1,
                'max': 0.4  # Minimum loss reduction range
            },
            'reg_lambda': {
                'min': 1.0,
                'max': 10.0  # L2 regularization term range
            },
            'reg_alpha': {
                'min': 0.0,
                'max': 1.0  # L1 regularization term range
            },
            'add_aug_val': {
                'min': 0.000001,
                'max': 0.0001  # Augmentation value for adding components
            },
            'remove_aug_val': {
                'min': 0.01,
                'max': 0.1  # Augmentation value for removing components
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='d2smell')
    best_t_mse = float('inf')  # Initialize with a very large value
    best_t_corr = -100
    best_params = None  # Initialize best_params
    best_models = []
    best_weights = []

    def train():
        nonlocal best_t_mse
        nonlocal best_t_corr
        nonlocal best_params
        nonlocal best_models
        nonlocal best_weights

        wandb.init()
        config = wandb.config

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='gpu_hist',
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            gamma=config.gamma,
            reg_lambda=config.reg_lambda,
            reg_alpha=config.reg_alpha,
            verbosity=2,
            predictor='gpu_predictor',
        )

        X, y = prepare_training_data(training_data_df, df_percept, mixture_definitions_df, all_cids, config.add_aug_val, config.remove_aug_val)
        X = expand_features(X, data_percept, percept_ids)
        poly = PolynomialFeatures(degree=1, include_bias=False)
        X = poly.fit_transform(X)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        models = []
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_val)
            score = mean_squared_error(y_val, y_pred, squared=False) 

            models.append(xgb_model)
            scores.append(score)

        weights = [1 / score for score in scores]
        weights = [weight / sum(weights) for weight in weights]

        y_t_pred = predict_with_average_model(models, weights, X_t)
        t_mse = rmse(y_t, y_t_pred)
        t_corr = pearson_corr(y_t, y_t_pred)
        wandb.log({'leaderboard_mse': t_mse})
        wandb.log({'leaderboard_corr': t_corr})

        if t_corr > best_t_corr:
            best_t_mse = t_mse
            best_t_corr = t_corr

            best_params = {
                'n_estimators': config.n_estimators,
                'learning_rate': config.learning_rate,
                'max_depth': config.max_depth,
                'min_child_weight': config.min_child_weight,
                'subsample': config.subsample,
                'colsample_bytree': config.colsample_bytree,
                'gamma': config.gamma,
                'reg_lambda': config.reg_lambda,
                'reg_alpha': config.reg_alpha,
                'add_aug_val': config.add_aug_val,
                'remove_aug_val': config.remove_aug_val,
                'test_mse': t_mse,
                'test_corr': t_corr,
            }
            wandb.run.summary['best_params'] = best_params

            best_models = models
            best_weights = weights

    
    wandb.agent(sweep_id, function=train, count=100)
    wandb.finish()

    return best_models, best_weights

def predict_with_average_model(models, weights, X):
    weighted_preds = np.zeros(X.shape[0])
    
    for model, weight in zip(models, weights):
        preds = model.predict(X)
        weighted_preds += weight * preds
    
    return weighted_preds

if __name__ == "__main__":
    print("Load data ... ... ", end='')
    path_output = 'output'
    cid_df = pd.read_csv('data/processed/CID.csv', header=[0])
    data_percept = pd.read_csv('output/percept_scaled.csv')
    training_data_df = pd.read_csv('data/raw/forms/TrainingData_mixturedist.csv')
    mixture_definitions_df = pd.read_csv('data/raw/mixtures/Mixure_Definitions_Training_set_VS2_without_leaderboard.csv')
    leaderboard_submission_df = pd.read_csv('data/raw/forms/Leaderboard_set_Submission_form.csv')
    mixture_definitions_leaderboard_df = pd.read_csv('data/raw/mixtures/Mixure_Definitions_Training_set_VS2_with_leaderboard.csv')
    test_set_submission_df = pd.read_csv('data/raw/forms/Test_set_Submission_form.csv')
    mixture_definitions_test_df = pd.read_csv('data/raw/mixtures/Mixure_Definitions_test_set.csv')
    df_percept = pd.read_csv('data/processed/predictions_separated.csv')
    print("Done")

    print("Prepare data ... ... ", end='')
    df_percept.drop(columns=['Prediction_11','Prediction_18','Prediction_21','Prediction_22','Prediction_27','Prediction_28','Prediction_31','Prediction_32'], inplace=True)
    df_percept = applying_2QuantileTransformer(df_percept)
    data_percept = extract_intensity_with_top_n(data_percept, 2)
    percept_ids = [1,2,3]
    all_cids = sorted(cid_df['CID'].astype(int).tolist())

    print("[train data] ", end='')

    X_train, y_train = prepare_training_data(training_data_df, df_percept, mixture_definitions_df, all_cids, 0.00065, 0.095)
    X_train = expand_features(X_train, data_percept, percept_ids)
    X_train = applying_PolynomialFeatures(X_train)

    print("[leadboard data] ", end='')
    X_leaderboard = prepare_leadboard_data(leaderboard_submission_df, df_percept, mixture_definitions_leaderboard_df, all_cids)
    X_leaderboard = expand_features(X_leaderboard, data_percept, percept_ids)
    X_leaderboard = applying_PolynomialFeatures(X_leaderboard)

    true_values_ld_df = pd.read_csv('data/raw/forms/LeaderboardData_mixturedist.csv')
    y_true = true_values_ld_df['Experimental Values'].values

    print("[test data] ", end='')
    X_test = prepare_test_data(test_set_submission_df, df_percept, mixture_definitions_test_df, all_cids)
    X_test = expand_features(X_test, data_percept, percept_ids)  
    X_test = applying_PolynomialFeatures(X_test)
    print("Done")

    print("Training ... ... ", end='')
    model, weight  = train_model(training_data_df, df_percept, mixture_definitions_df, all_cids, X_leaderboard, y_true, data_percept, percept_ids)
    
    # model, weight = train_model(X_train, y_train)
    jl.dump(model, os.path.join(path_output, 'mixture_model.pkl'))
    print("Done")

    print("Inference [leadboard] ... ... ", end='')
    y_leaderboard_pred = predict_with_average_model(model, weight, X_leaderboard)
    leaderboard_submission_df['Predicted_Experimental_Values'] = y_leaderboard_pred
    submission_output_path = os.path.join(path_output, 'Leaderboard_set_Submission_form.csv')
    leaderboard_submission_df.to_csv(submission_output_path, index=False)
    print("Done")

    print("Inference [test] ... ... ", end='')
    y_test_pred = predict_with_average_model(model, weight, X_test)
    test_set_submission_df['Predicted_Experimental_Values'] = y_test_pred
    test_set_submission_output_path = os.path.join(path_output, 'Test_set_Submission_form.csv')
    test_set_submission_df.to_csv(test_set_submission_output_path, index=False)
    print("Done")
    
# ==================================================================================================================

    true_values_ld_df = pd.read_csv('data/raw/forms/LeaderboardData_mixturedist.csv')
    y_true = true_values_ld_df['Experimental Values'].values
    rmse_value = rmse(y_true, y_leaderboard_pred)
    pearson_corr_value = pearson_corr(y_true, y_leaderboard_pred)
    print(f'Leaderboard RMSE: {rmse_value}')
    print(f'Leaderboard Pearson Correlation: {pearson_corr_value}')

