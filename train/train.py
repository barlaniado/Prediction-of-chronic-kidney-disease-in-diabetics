import sys
import os
from sklearn.ensemble import AdaBoostClassifier
from utilities_train.preprocessing_dataset_for_train import preprocessing, get_features_and_target
import pickle

DATA_PATH = '../../../diab_ckd_data.CSV'


def fit_save_model(data_path):
    data = get_features_and_target(preprocessing(data_path))
    ada_clf = AdaBoostClassifier(learning_rate=1.2, n_estimators=500)
    ada_clf.fit(data[0], data[1])
    with open('../model/ada_clf', 'wb') as file:
        pickle.dump(ada_clf, file)








