import joblib
import pandas as pd
from flask import request, Flask
import feature_names_and_types


pred = Flask(__name__)

with open('../model/model.joblib', 'rb') as f:
    MODEL = joblib.load(f)

FEATURES = feature_names_and_types.FEATURES

@pred.route('/predict_crf')
def return_prediction():
    to_pred = pd.DataFrame({key: request.args.get(key) for key in FEATURES.keys()},
                 index=[0])
    for c in to_pred.columns:
        to_pred[c] = to_pred[c].astype(FEATURES[c])
    return str(Predict(MODEL, to_pred).predictions[0][-1])

class Predict:
    def __init__(self, model, X):
        self.model = model
        self.predict(X)

    def predict(self, X):
        self.predictions = self.model.predict_proba(X)


if __name__ == '__main__':
    pred.run()
