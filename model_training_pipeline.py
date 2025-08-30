from src.pipelines import training_data_processing_pipeline, testing_data_processing_pipeline
from src.train_model import ModelTraining
import pandas as pd
import numpy as np
import json

class ModelTrainingPipeline:
    def __init__(self, epochs=5, batch_size=64, validation_split=0.3):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        

    def load_data(self):
        """Load processed training and testing data from JSON files."""
        with open("data\\processed_data\\testing_data.json", 'r') as file:
            data = json.load(file)
            test_X = np.array(data['test_X'])
            test_y = np.array(data['test_y'])

        with open("data\\processed_data\\training_data.json", 'r') as file:
            data = json.load(file)
            train_X = np.array(data['train_X'])
            train_y = np.array(data['train_y'])
        return train_X, train_y, test_X, test_y


    def train_model(self):
        train_X, train_y, test_X, test_y = self.load_data()
        print(train_X.shape, train_y.shape)
        print(test_X.shape, test_y.shape)

        model_trainer = ModelTraining(timestamps=train_X.shape[1], features=train_X.shape[2])
        model = model_trainer.fit(
                                  X_train = train_X, y_train = train_y, 
                                  epochs=self.epochs, batch_size = self.batch_size, 
                                  validation_split = self.validation_split
                                  )
        return model