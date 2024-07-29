import itertools
import joblib as jl
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PolynomialFeatures

predictions_separated_num = 32

def fit_and_standardize(data):
    scaler = StandardScaler().fit(data)
    scaled = scaler.transform(data)

    return scaled, scaler


def fit_and_standardize_minmax(data):
    scaler = MinMaxScaler().fit(data)
    scaled = scaler.transform(data)

    return scaled, scaler


def standardize(data, scaler):
    scaled = scaler.transform(data)

    return scaled


def inv_standardize(data, scaler):      
    scaled = scaler.inverse_transform(data)
    return scaled


def load_scaler(scaler_path):
    scaler = jl.load(scaler_path)
    return scaler


def extract_intensity_with_top_n(df, n):
    result_data = []
    excluded_columns = ['INTENSITY', 'PLEASANTNESS', 'CID']

    for index, row in df.iterrows():
        intensity_value = row['INTENSITY']

        filtered_row = row.drop(excluded_columns)
        top_two_values = filtered_row.nlargest(n)
        top_two_indices = [df.columns.get_loc(col) for col in top_two_values.index]

        entry = [row['CID'], intensity_value]
        for i, (index, value) in enumerate(zip(top_two_indices, top_two_values)):
            entry.append(index)
            entry.append(value)
        
        result_data.append(entry)
    
    column_labels = ['CID', 'INTENSITY_Value'] + ['Top_{}_Col_Index'.format(i+1) for i in range(n)] + ['Top_{}_Value'.format(i+1) for i in range(n)]
    result_df = pd.DataFrame(result_data, columns=column_labels)
    return result_df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pearson_corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def prepare_binary_vector(mixture_definitions_df, all_cids, mixture_id):
    binary_vector = np.zeros(len(all_cids))
    mixture_row = mixture_definitions_df[mixture_definitions_df['Mixture Label'] == mixture_id].iloc[:, 2:].values.flatten()
    mixture_components = mixture_row[~pd.isna(mixture_row)].astype(int)
    for cid in mixture_components:
        if cid in all_cids:
            binary_vector[all_cids.index(cid)] = 1
    return binary_vector


def data_aug_add(mix1, mix2, exp_val, aug_val, aug_num_list):
    new_mix1_list = []
    new_exp_val_list = []
    idx = np.where((mix2 == 1) & (mix1 == 0))[0]

    for aug_num in aug_num_list:
        idx_combinations = list(itertools.combinations(idx, aug_num))
        
        for combination in idx_combinations:
            new_mix1 = mix1.copy()
            new_exp_val = exp_val - aug_val * aug_num
            
            for i in combination:
                new_mix1[i] = 1
            
            new_mix1_list.append(new_mix1)
            new_exp_val_list.append(new_exp_val)
    
    return new_mix1_list, new_exp_val_list


def data_aug_remove(mix1, mix2, exp_val, aug_val, aug_num_list):
    new_mix1_list = []
    new_mix2_list = []
    new_exp_val_list = []

    idx = np.where((mix1 == 1) & (mix2 == 1))[0]

    for aug_num in aug_num_list:
        idx_combinations = list(itertools.combinations(idx, aug_num))
        
        for combination in idx_combinations:
            new_mix1 = mix1.copy()
            new_mix2 = mix2.copy()
            new_exp_val = exp_val + aug_val * aug_num
            
            for i in combination:
                new_mix1[i] = 0
                new_mix2[i] = 0
            
            new_mix1_list.append(new_mix1)
            new_mix2_list.append(new_mix2)
            new_exp_val_list.append(new_exp_val)
    
    return new_mix1_list, new_mix2_list, new_exp_val_list


def expand_features(X, feature_matrix, ids):

    pred_features_length = predictions_separated_num*2
    X_p = X[:, -pred_features_length:]
    X = X[:, :-pred_features_length]

    features = feature_matrix.values
    features = features[:,ids]
    
    n_samples, n_features = X.shape
    n_id = int(n_features/2)
    
    id_length = len(ids)

    new_n_features = n_features + n_features * id_length
    expanded_X = np.zeros((n_samples, new_n_features))
    
    expanded_X[:, :n_features] = X
    
    for j in range(n_id):
        indices = np.where(X[:, j] == 1)[0]
        expanded_X[indices, n_features + j * id_length: n_features + (j + 1) * id_length] = features[j]

    for j in range(n_id):
        indices = np.where(X[:, n_id + j] == 1)[0]
        expanded_X[indices, n_features + n_id * id_length + j * id_length: n_features + n_id * id_length + (j + 1) * id_length] = features[j]
    
    expanded_X = np.concatenate([expanded_X, X_p], axis=1)
    return expanded_X


def applying_2QuantileTransformer(df):
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=500, random_state=0)
    prediction_columns = [col for col in df.columns if col.startswith('Prediction_')]
    df[prediction_columns] = scaler.fit_transform(scaler.fit_transform(df[prediction_columns]))
    return df


def applying_PolynomialFeatures(x, degree=1):
    Polynomialdegree = degree
    poly = PolynomialFeatures(degree=Polynomialdegree, include_bias=False)
    x = poly.fit_transform(x)

    return x