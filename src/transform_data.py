from sklearn.pipeline import Pipeline
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DropColumns:
    """
    Transformer to drop specified columns from a DataFrame and create an 'engine_id' column.

    Args:
        columns (list): List of column names to drop.
    """
    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DropColumns':
        """
        Adds 'engine_id' column to the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            DropColumns: self
        """
        X['engine_id'] = X['file_name'] + '_' + X['0'].astype(str)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return X.drop(columns=self.columns, axis='columns')


class RULProcessing:
    """
    Transformer to generate Remaining Useful Life (RUL) for training or testing data.

    Args:
        typ_data (str): Type of data, either 'train' or 'test'.
        rul_df (pd.DataFrame, optional): DataFrame containing RUL values for test data.
    """
    def __init__(self, typ_data: str, rul_df: pd.DataFrame = None):
        self.typ_data = typ_data
        self.rul_df = rul_df

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'RULProcessing':
        """
        Fit method (does nothing).

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            RULProcessing: self
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates RUL for training or testing data.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with RUL column added.
        """
        if self.typ_data == 'train':
            generated_rul = []
            for engine_id, cycles in dict(X['engine_id'].value_counts()).items():
                temp = list(range(1, cycles+1))
                temp.reverse()
                generated_rul += temp
        else:
            generated_rul = []
            for rul, (engine_id, cycles) in zip(self.rul_df['0'], dict(X['engine_id'].value_counts()).items()):
                temp = list(range(1, cycles + 1 + int(rul)))
                temp.reverse()
                generated_rul += temp[:cycles]

        X['RUL'] = generated_rul
        return X
    

class NormalizeData:
    """
    Transformer to normalize numerical columns and encode categorical columns.

    Args:
        cat_cols (list): List of categorical column names.
        exclude_num_cols (list): List of column names to exclude from normalization.
    """
    minmax_scaler = MinMaxScaler()
    label_encoder = LabelEncoder()

    def __init__(self, cat_cols: list, exclude_num_cols: list):
        self.cat_cols = cat_cols
        self.exclude_num_cols = exclude_num_cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'NormalizeData':
        """
        Fits the scaler and encoder.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            NormalizeData: self
        """
        self.minmax_cols = X.drop(self.exclude_num_cols, axis='columns').columns
        self.minmax_scaler.fit(X[self.minmax_cols])
        for col in self.cat_cols:
            self.label_encoder.fit(X[col].astype(str).values.ravel())
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes numerical and categorical columns.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        for col in self.cat_cols:
            X[col] = self.label_encoder.transform(X[col].astype(str).values.ravel())
        X[self.minmax_cols] = self.minmax_scaler.transform(X.drop(self.exclude_num_cols, axis='columns'))
        return X
    
    
class SequentialDataProcessing:
    """
    Transformer to split data into sequential chunks for LSTM model training.

    Args:
        sequence_length (int): Length of each sequence window.
    """
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'SequentialDataProcessing':
        """
        Fit method (does nothing).

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            SequentialDataProcessing: self
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Splits data into chunks to create training data for LSTM model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Sequentially processed DataFrame.
        """
        wind_X, wind_y = [], []
        for i, (engine, cycles) in enumerate(dict(X['engine_id'].value_counts(sort=True)).items()):
            sys.stdout.flush()
            print('applying window for engine number = ', i + 1, " out of ", len(X['engine_id'].unique()), end="\r")
            train_temp = X.drop(['RUL'], axis='columns').astype('float16')
            for i in range(cycles-10):
                
                train_x_temp = np.array(train_temp.iloc[i:i+10])
                train_y_temp = X['RUL'].iloc[i + 10]

                wind_X.append(train_x_temp)
                wind_y.append(train_y_temp)

        return np.array(wind_X), np.array(wind_y)
    

if __name__ == "__main__":

    import config
    train_raw_data = pd.read_csv(config.RAW_TRAIN_PATH)
    test_raw_data = pd.read_csv(config.RAW_TEST_PATH)
    rul_raw_data = pd.read_csv(config.RAW_RUL_PATH)
    print(rul_raw_data.head())

    training_pipeline = Pipeline(steps=[
        ('drop_columns', DropColumns(columns=['26', '27', 'file_name'])),
        ('rul_processing', RULProcessing(typ_data='train')),
        ('normalize_data', NormalizeData(cat_cols=['engine_id'], exclude_num_cols=config.EXCLUDE_MINMAX_COLS)),
        ('sequential_data_processing', SequentialDataProcessing(sequence_length=19))
    ])


    testing_pipeline = Pipeline(steps=[
        ('drop_columns', DropColumns(columns=['26', '27', 'file_name'])),
        ('rul_processing', RULProcessing(typ_data='test', rul_df=rul_raw_data)),
        ('normalize_data', NormalizeData(cat_cols=['engine_id'], exclude_num_cols=config.EXCLUDE_MINMAX_COLS)),
        ('sequential_data_processing', SequentialDataProcessing(sequence_length=19))
    ])


    train_X, train_y = training_pipeline.fit_transform(X=train_raw_data, y=None)
    # test_X, test_y = testing_pipeline.fit_transform(X=test_raw_data, y=None)
    
    print(train_X.shape, train_y.shape)
    # print(test_X.shape, test_y.shape)


    