import json
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle


def read_dataset(path):
    data = pd.read_csv(path)
    return data


def read_json(path):
    with open(path) as json_file:
        file = json.load(json_file)
    return file


def create_model_to_train(estimator):
    """" create  classifer object with the hyper-parameters we found in our research """
    model_to_train = estimator
    return model_to_train


def create_preprocessing_pipeline(path_how_to_handle_columns):
    """ build pipeline for preprocessing the data """

    # save the column names to variables
    json_file_handle_col = read_json(path_how_to_handle_columns)
    cat_cols = json_file_handle_col['cat_cols']
    impute_mean_cols = json_file_handle_col['impute_mean_cols']
    impute_zero_cols = json_file_handle_col['impute_zero_cols']

    # create an object of ColumnTransformer
    pre_process_pipeline = ColumnTransformer([
        ('cat_cols', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('impute_mean_cols', SimpleImputer(strategy='mean'), impute_mean_cols),
        ('impute_zero_cols', SimpleImputer(strategy='constant', fill_value=0), impute_zero_cols),
    ], remainder='drop')
    return pre_process_pipeline


def create_whole_pipeline(path_how_to_handle_columns='./configuration_train/handle_columns.json',
                          estimator=AdaBoostClassifier(learning_rate=1.2, n_estimators=500)):
    pre_process_pipeline = create_preprocessing_pipeline(path_how_to_handle_columns)
    cls = create_model_to_train(estimator)
    whole_pipeline = Pipeline([('pre-process', pre_process_pipeline),
                      ('estimator', cls)])
    return whole_pipeline


def preprocessing_target_variable(data):
    data.loc[data['TIME_CRF'] > 10, 'EVENT_CRF'] = 0
    data = data.loc[~((data['TIME_CRF'] < 10) & (data['EVENT_CRF'] == 0))]
    return data


def get_features_and_target(data, json_features_and_target='./configuration_train/features_and_target.json'):
    json_file = read_json(json_features_and_target)
    features = json_file['features']
    target = json_file['target']
    X = data[features]
    y = data[target]
    return X, y




