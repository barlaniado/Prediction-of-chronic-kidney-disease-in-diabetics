from utilities_train import preprocessing_dataset_for_train as pre
import joblib


DATA_PATH = '../../../diab_ckd_data.CSV'


def fit_save_model(data_path):
    # Load the data
    data = pre.read_dataset(DATA_PATH)
    data = pre.preprocessing_target_variable(data)
    X, y = pre.get_features_and_target(data)
    pipeline = pre.create_whole_pipeline()
    pipeline.fit(X, y)

    # save the whole fitted pipeline
    with open('../model/model.joblib', 'wb') as f:
        joblib.dump(pipeline, f, compress='zlib')


fit_save_model(DATA_PATH)




