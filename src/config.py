import os

os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.getcwd(), "data")
METADATA = ['readme.txt', 'x.txt', 'Damage Propagation Modeling.pdf']
RAW_DATA_PATH = os.path.join(os.getcwd(), "data", "raw_data")
EXCLUDE_MINMAX_COLS = ['engine_id', 'RUL', '0', '1']


RAW_TRAIN_PATH = os.path.join(os.getcwd(), "data", "raw_data", "train_data.csv")
RAW_TEST_PATH = os.path.join(os.getcwd(), "data", "raw_data", "test_data.csv")
RAW_RUL_PATH = os.path.join(os.getcwd(), "data", "raw_data", "rul_data.csv")
PROCESSED_TRAIN_PATH = os.path.join(os.getcwd(), "data", "processed_data", "train.csv")
PROCESSED_TEST_PATH = os.path.join(os.getcwd(), "data", "processed_data", "train.csv")