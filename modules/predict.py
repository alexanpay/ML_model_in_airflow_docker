import pandas as pd
import dill
import os
import json

from datetime import datetime
from pydantic import BaseModel


with open(f'\\\\wsl$\\Ubuntu-22.04\\home\\alexa\\airflow_hw\\data\\models\\cars_pipe_202311030318.pkl',
          'rb') as file:
    model = dill.load(file)

path_to_files = '\\\\wsl$\\Ubuntu-22.04\\home\\alexa\\airflow_hw\\data\\test'
files = os.listdir(path_to_files)


class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


'''def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)'''
def predict(form: Form):
    df = pd.DataFrame([form.model_dump()])
    return model['model'].predict(df)

if __name__ == '__main__':
    predictions = {}
    for json_file in files:
        file_path = os.path.join(path_to_files, json_file)

        with open(file_path, 'r') as file:
            data = json.load(file)
            form_instance = Form(**data)
            y = predict(form_instance)
            predictions.update({data['id']: y[0]})

    final_directory = '\\\\wsl$\\Ubuntu-22.04\\home\\alexa\\airflow_hw\\data\\predictions'
    final_path = os.path.join(final_directory, f'predictions_{datetime.now().strftime("%Y%m%d%H%M")}')

    with open(final_path, 'w') as json_f:
        json.dump(predictions, json_f)


