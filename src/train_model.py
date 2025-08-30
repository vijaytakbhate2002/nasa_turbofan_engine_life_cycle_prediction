import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPooling1D, Conv1D

class ModelTraining:
    def __init__(self, timestamps: int, features: int, pool_size: int = 2, filters: int = 64, kernel_size: int = 3):
        self.timestamps = timestamps
        self.features = features
        self.pool_size = pool_size
        self.filters = filters
        self.kernel_size = kernel_size

    def buildModel(self) -> Sequential:
        """Builds and returns a Sequential model."""

        model = Sequential(
            [
                Conv1D(filters = self.filters, kernel_size = self.kernel_size, activation='relu', input_shape = (self.timestamps, self.features)),
                MaxPooling1D(pool_size = self.pool_size),

                LSTM(64, return_sequences = False),

                Dense(self.filters, activation = 'relu'),
                Dense(1)

            ]
        )
        return model
    

    def compileModel(self) -> None:
        """Compiles the model with Adam optimizer and Mean Squared Error loss."""
        model = self.buildModel()
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(model.summary())
        return model


    def fit(self, X_train, y_train, epochs:int=30, batch_size:int=64, validation_split:float=0.3) -> Sequential:
        """Fits the model on training data."""
        model = self.compileModel()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return model
    

if __name__ == "__main__":
    from pipelines import training_data_processing_pipeline, testing_data_processing_pipeline
    import config
    import pandas as pd
    import json

    with open("E:\\WorkSpace\\Complete_MLOps_Bootcamp_With_10_End_To_End_ML_Projects\\nasa_turbofan_engine_RUL_prediction\\turbofan_engine_life_prediction\\data\\processed_data\\testing_data.json", 'r') as file:
        data = json.load(file)
        test_X = data['test_X'].tonumpy()
        test_y = data['test_y'].tonumpy()

    with open("E:\\WorkSpace\\Complete_MLOps_Bootcamp_With_10_End_To_End_ML_Projects\\nasa_turbofan_engine_RUL_prediction\\turbofan_engine_life_prediction\\data\\processed_data\\training_data.json", 'r') as file:
        import json
        data = json.load(file)
        train_X = data['train_X'].tonumpy()
        train_y = data['train_y'].tonumpy()

    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)

    model_trainer = ModelTraining(timestamps=train_X.shape[1], features=train_X.shape[2])
    model = model_trainer.fit(X_train=train_X, y_train=train_y, epochs=5, batch_size=64, validation_split=0.3)
        