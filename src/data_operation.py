import kagglehub
import logging
import os
import config
import pandas as pd

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataOperation:

    data_path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    data_path = os.path.join(data_path, "CMaps")
    logging.info(f"Data downloaded from {data_path}")
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    rul_df = pd.DataFrame()

    def extractData(self) -> None:
        """Extracts and load data into the raw data directory (train, test, rul)"""
        try:
            file_names = []
            for file in os.listdir(self.data_path):
                if file not in config.METADATA:
                    file_names.append(file)
            logging.info(f"Data files found: {file_names}")

            for file in file_names:
                if 'train' in file:
                    df = pd.read_csv(os.path.join(self.data_path, file), sep=' ', header=None)
                    df['file_name'] = file
                    self.train_df = pd.concat([self.train_df, df], axis=0)
                elif 'test' in file:
                    df = pd.read_csv(os.path.join(self.data_path, file), sep=' ', header=None)
                    df['file_name'] = file
                    self.test_df = pd.concat([self.test_df, df], axis=0)
                elif 'RUL' in file:
                    df =  pd.read_csv(os.path.join(self.data_path, file), sep=' ', header=None)
                    self.rul_df['file_name'] = file
                    self.rul_df = pd.concat([self.rul_df, df], axis=0)

            self.train_df.to_csv(os.path.join(config.RAW_DATA_PATH, 'train_data.csv'), index=False)
            self.test_df.to_csv(os.path.join(config.RAW_DATA_PATH, 'test_data.csv'), index=False)
            self.rul_df.to_csv(os.path.join(config.RAW_DATA_PATH, 'rul_data.csv'), index=False)
            logging.info(f"Data loaded in {config.RAW_DATA_PATH} successfully")

        except Exception as e:
            logging.error(f"Error loading data: {e}")        

    # def dumpProcessedData(self, df: pd.DataFrame, file_name: str) -> None:
    #     """Dumps the processed data into the processed data directory"""
    #     try:
    #         processed_data_path = os.path.join(os.getcwd(), "data", "processed_data")
    #         if not os.path.exists(processed_data_path):
    #             os.makedirs(processed_data_path)
    #         df.to_csv(os.path.join(processed_data_path, file_name), index=False)
    #         logging.info(f"Processed data dumped in {processed_data_path} as {file_name}")
    #     except Exception as e:
    #         logging.error(f"Error dumping processed data: {e}")

if __name__ == "__main__":
    data_op = DataOperation()
    data_op.extractData()
    