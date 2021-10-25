from utilities_train import preprocessing_dataset_for_train as pre
import joblib


DATA_PATH = '../../../diab_ckd_data.CSV'


def fit_save_model(data_path):
    # Load the data
    data = pre.read_dataset(DATA_PATH)
    data = pre.preprocessing_target_variable(data)
    X, y = pre.get_features_and_target(data)
    pipline = pre.create_whole_pipeline()
    pipline.fit(X, y)
    # save the whole fitted pipeline
    joblib.dump(pipline, '../model/model.pkl', compress=1)







