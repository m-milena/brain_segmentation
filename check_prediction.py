import requests
import zlib
import numpy as np
import nibabel as nib

from typing import Tuple, List
from pathlib import Path
import os

list_of_files = [f for f in os.listdir('.') if f[-2:] == 'gz']

for prediction_path in list_of_files:
    prediction_name = prediction_path[:-7]
    prediction = nib.load(prediction_path)

    response = requests.post(f'http://vision.dpieczynski.pl:8080/{prediction_name}', data=zlib.compress(prediction.to_bytes()))
    if response.status_code == 200:
        print(prediction_path, prediction_name, response.json())
        with open('results.txt', 'a') as f:
            f.write(prediction_name+','+ str(response.json()) + '\n')
    else:
        print(f'Error processing prediction {dataset_predictions_path.name}/{prediction_name}: {response.text}')
