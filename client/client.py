import requests
import json

URL = 'http://127.0.0.1:5000/predict_crf'


class ObservationToPredict:
    def __init__(self, path_json_x):
        with open(path_json_x) as x:
            self.data = json.load(x)

        self.get_prediction()

    def get_prediction(self):
        self.prediction = float(requests.get(URL,
                                             params=self.data).text)

    def __str__(self):
        string_to_print = f'According to the model,' \
                          f' the probability of the diabetic to' \
                          f' develop chronic kidney disease in' \
                          f' the next 10 years is: {round(self.prediction, 3)}'
        return string_to_print


if __name__ == '__main__':
    print(ObservationToPredict('./sample.json'))
