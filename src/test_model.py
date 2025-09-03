import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
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



def test_model(model, test_X, test_y):
    """
    Evaluates the model and returns loss, MAE, and accuracy.

    Args:
        model: Trained model.
        test_X: Test features.
        test_y: Test labels.

    Returns:
        tuple: (loss, mae, accuracy)
    """
    y_pred = model.predict(test_X).flatten()
    y_pred = list(map(int, y_pred))
    difference_abs = [abs(a - b) for a, b in zip(y_pred, test_y)]
    difference_sqr = [(a - b)**2 for a, b in zip(y_pred, test_y)]
    mae = np.mean(difference_abs)
    mse = np.mean(difference_sqr)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_y, y_pred)

    return mae, mse, rmse, r2


if __name__ == "__main__":
    temp = ModelTrainingPipeline()

    train_X, train_y, test_X, test_y = temp.load_data()
    print(train_X.shape, train_y.shape)
    print(test_model(None, train_X, train_y))