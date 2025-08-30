from src.pipelines import training_data_processing_pipeline, testing_data_processing_pipeline
from src import config
import pandas as pd
import json
import os

train_raw_data = pd.read_csv(config.RAW_TRAIN_PATH)
test_raw_data = pd.read_csv(config.RAW_TEST_PATH)
rul_raw_data = pd.read_csv(config.RAW_RUL_PATH)


train_X, train_y = training_data_processing_pipeline.fit_transform(X=train_raw_data, y=None)
test_X, test_y = testing_data_processing_pipeline.fit_transform(X=test_raw_data, y=None)


os.makedirs("data\\processed_data", exist_ok=True)
if not os.path.exists("data\\processed_data\\testing_data.json"):
    with open("data\\processed_data\\testing_data.json", 'w') as file:
        json.dump({'test_X':[], 'test_y':[]}, file)
if not os.path.exists("data\\processed_data\\training_data.json"):
    with open("data\\processed_data\\training_data.json", 'w') as file:
        json.dump({'trainX':[], 'train_y':[]}, file)



with open("data\\processed_data\\testing_data.json", 'r') as file:
    data = json.load(file)
    data['test_X'] = test_X.tolist()
    data['test_y'] = test_y.tolist()

with open("data\\processed_data\\testing_data.json", 'w') as file:
    json.dump(data, file)
    


with open("data\\processed_data\\training_data.json", 'r') as file:
    data = json.load(file)
    data['train_X'] = train_X.tolist()
    data['train_y'] = train_y.tolist()

with open("data\\processed_data\\training_data.json", 'w') as file:
    json.dump(data, file)