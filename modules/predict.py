import json
import os
from datetime import datetime

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    best_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{best_model}', 'rb') as file:
        model = dill.load(file)

    path_to_files = f'{path}/data/test'
    files = os.listdir(path_to_files)

    predictions = {}
    for json_file in files:
        file_path = os.path.join(path_to_files, json_file)

        with open(file_path, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame([data])
            y = model['model'].predict(df)
            predictions.update({data['id']: y[0]})

    final_path = f'{path}/data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}'

    with open(final_path, 'w') as json_f:
        json.dump(predictions, json_f)


if __name__ == '__main__':
    predict()