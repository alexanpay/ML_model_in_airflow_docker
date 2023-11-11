import json
import os
from datetime import datetime

import dill
import pandas as pd

with open(f'~/airflow/airflow_hw/data/models/cars_pipe_202311030318.pkl', 'rb') as file:
    model = dill.load(file)

path_to_files = '~/airflow/airflow_hw/data/test'
files = os.listdir(path_to_files)


def predict():
    predictions = {}
    for json_file in files:
        file_path = os.path.join(path_to_files, json_file)

        with open(file_path, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame([data])
            y = model['model'].predict(df)
            predictions.update({data['id']: y[0]})

    final_directory = '~/airflow/airflow_hw/data/predictions'
    final_path = os.path.join(final_directory, f'predictions_{datetime.now().strftime("%Y%m%d%H%M")}')

    with open(final_path, 'w') as json_f:
        json.dump(predictions, json_f)


if __name__ == '__main__':
    predict()