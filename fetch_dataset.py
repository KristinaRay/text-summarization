import os
from secret import *

os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print(f'Downloading the dataset..')
os.makedirs('dataset', exist_ok=True)
print("Dataset not found, downloading...")
args = {'force': True, 'quiet': False, 'unzip': True}

# Download all files of a dataset 

api.dataset_download_files('gowrishankarp/newspaper-text-summarization-cnn-dailymail', 'dataset', **args)