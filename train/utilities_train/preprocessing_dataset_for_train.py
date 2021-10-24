import json
import pandas as pd
import pickle



def read_dataset(path):
    data = pd.read_csv(path)
    return data


def read_json(path):
    with open(path) as json_file:
        file = json.load(json_file)
    return file


def fill_missing_values(data, path_constant_file):
    values_to_fill = read_json(path_constant_file)
    return data.fillna(values_to_fill)


def one_hot_encoder(data, encoder_path, columns_to_encode_file_json):
    columns_to_encode = read_json(columns_to_encode_file_json)
    encoder = pickle.load(open(encoder_path, 'rb'))
    data_cat_enc = pd.DataFrame(encoder.transform(data[columns_to_encode]).toarray().astype('int'),
                                columns=encoder.get_feature_names(columns_to_encode),
                                index=data.index)
    data_no_cat = data.drop(axis=1, labels=columns_to_encode)
    data = pd.concat([data_no_cat, data_cat_enc], axis=1)
    return data


def preprocessing_target_variable(data):
    data.loc[data['TIME_CRF'] > 10, 'EVENT_CRF'] = 0
    data = data.loc[~((data['TIME_CRF'] < 10) & (data['EVENT_CRF'] == 0))]
    return data


def preprocessing(data_path, missing_values_file='./configuration_train/fill_missing_values_constants.json',
                  encoder_path='./utilities_train/one_hot_encoder',
                  columns_to_encode_file='./configuration_train/columns_to_encode.json'):
    data_to_process = read_dataset(data_path)
    data_to_process = fill_missing_values(data_to_process, missing_values_file)
    try:
        assert data_to_process.isna().sum().sum() == 0
    except AssertionError:
        raise AssertionError('Not all missing values were filled')
    data_to_process = one_hot_encoder(data_to_process, encoder_path, columns_to_encode_file)
    data_to_process = preprocessing_target_variable(data_to_process)
    return data_to_process


def get_features_and_target(data, json_features_and_target='./configuration_train/final_features_and_target.json'):
    json_file = read_json(json_features_and_target)
    features = json_file['final_features']
    target = json_file['target']
    X = data[features]
    y = data[target]
    return X, y