import joblib
import pandas as pd
import numpy
from flask import request, Flask

pred = Flask(__name__)

with open('../model/model.joblib', 'rb') as f:
    MODEL = joblib.load(f)

FEATURES = {'IS_MALE': numpy.int64,
             'AGE_AT_SDATE': numpy.float64,
             'AGE_GROUP': numpy.object_,
             'SES_GROUP': numpy.object_,
             'MIGZAR': numpy.object_,
             'IS_HYPERTENSION': numpy.int64,
             'SE_HYPERTENSION': numpy.float64,
             'IS_ISCHEMIC_MI': numpy.int64,
             'SE_ISCHEMIC_MI': numpy.float64,
             'IS_CVA_TIA': numpy.int64,
             'SE_CVA_TIA': numpy.float64,
             'IS_DEMENTIA': numpy.int64,
             'SE_DEMENTIA': numpy.float64,
             'IS_ART_SCLE_GEN': numpy.int64,
             'SE_ART_SCLE_GEN': numpy.float64,
             'IS_TROMBOPHILIA': numpy.int64,
             'SE_TROMBOPHILIA': numpy.float64,
             'IS_IBD': numpy.int64,
             'SE_IBD': numpy.float64,
             'BMI_AT_BASELINE': numpy.float64,
             'SYSTOLA_AT_BASELINE': numpy.float64,
             'DIASTOLA_AT_BASELINE': numpy.float64,
             'Creatinine_B_AT_BASELINE': numpy.float64,
             'Albumin_B_AT_BASELINE': numpy.float64,
             'Urea_B_AT_BASELINE': numpy.float64,
             'Glucose_B_AT_BASELINE': numpy.float64,
             'HbA1C_AT_BASELINE': numpy.float64,
             'RBCRed_Blood_Cells_AT_BASELINE': numpy.float64,
             'Hemoglobin_AT_BASELINE': numpy.float64,
             'Ferritin_AT_BASELINE': numpy.float64,
             'AST_GOT_AT_BASELINE': numpy.float64,
             'ALT_GPT_AT_BASELINE': numpy.float64,
             'Bilirubin_Total_AT_BASELINE': numpy.float64,
             'Na_Sodium_B_AT_BASELINE': numpy.float64,
             'K_Potassium_B_AT_BASELINE': numpy.float64,
             'CaCalcium_B_AT_BASELINE': numpy.float64,
             'HDLCholesterol_AT_BASELINE': numpy.float64,
             'LDLCholesterol_AT_BASELINE': numpy.float64,
             'Triglycerides_AT_BASELINE': numpy.float64,
             'PTH_AT_BASELINE': numpy.float64}

@pred.route('/')
def return_hello():
    return 'hello'


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
