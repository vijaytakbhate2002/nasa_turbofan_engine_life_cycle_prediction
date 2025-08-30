import pandas as pd
from sklearn.pipeline import Pipeline
from .transform_data import DropColumns, RULProcessing, NormalizeData, SequentialDataProcessing
from . import config

rul_raw_data = pd.read_csv(config.RAW_RUL_PATH)

training_data_processing_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumns(columns=['26', '27', 'file_name'])),
    ('rul_processing', RULProcessing(typ_data='train')),
    ('normalize_data', NormalizeData(cat_cols=['engine_id'], exclude_num_cols=config.EXCLUDE_MINMAX_COLS)),
    ('sequential_data_processing', SequentialDataProcessing(sequence_length=19))
])


testing_data_processing_pipeline = Pipeline(steps=[
    ('drop_columns', DropColumns(columns=['26', '27', 'file_name'])),
    ('rul_processing', RULProcessing(typ_data='test', rul_df=rul_raw_data)),
    ('normalize_data', NormalizeData(cat_cols=['engine_id'], exclude_num_cols=config.EXCLUDE_MINMAX_COLS)),
    ('sequential_data_processing', SequentialDataProcessing(sequence_length=19))
])
