from model_training_pipeline import ModelTrainingPipeline
from src.test_model import test_model
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
import sys
import os
import numpy as np
from tensorflow.keras.models import Sequential

# os.environ['MLFLOW_TRACKING_USERNAME'] = 'TYPE YOUR USERNAME HERE'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'TYPE YOUR PASSWORD HERE'

os.environ['MLFLOW_TRACKING_URI'] = 'http://ec2-3-81-87-234.compute-1.amazonaws.com:5000/'
os.environ['MLFLOW_TRACKING_URI'] = "mlruns"

URI = os.environ.get('MLFLOW_TRACKING_URI')

print(URI)
print( urlparse(URI).scheme)

mlflow.set_tracking_uri(URI)
mlflow.set_experiment("Turbofan Engine RUL Prediction")

with mlflow.start_run():

    if len(sys.argv) > 1:
        print("Arguments passed: ", sys.argv)
        epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        validation_split = float(sys.argv[3])
    else:
        epochs = 5
        batch_size = 64
        validation_split = 0.3
        
    trainer = ModelTrainingPipeline(
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split
        )
    
    train_X, train_y, test_X, test_y = trainer.load_data()

    print("Input Data View")
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print(train_X[:10], train_y[:10])
    print(test_X[:10], test_y[:10])

    model = trainer.train_model()

    y_pred = model.predict(test_X).reshape(1, -1)[0]

    test_mae, test_mse, test_rmse, test_r2 = test_model(model, test_X, test_y)
    train_mae, train_mse, train_rmse, train_r2 = test_model(model, train_X, train_y)

    y_pred = model.predict(test_X).flatten()
    y_pred = np.array(list(map(int, y_pred)))

    print(y_pred.shape, test_y.shape)
    print(y_pred[:30], test_y[:30])

    y_pred_train = model.predict(train_X).flatten()
    y_pred_train = np.array(list(map(int, y_pred_train)))

    print(y_pred_train.shape, train_y.shape)
    print(y_pred_train[:30], train_y[:30])

    print(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test RMSE: {test_rmse}, Test R2: {test_r2}")
    print(f"Train MAE: {train_mae}, Train MSE: {train_mse}, Train RMSE: {train_rmse}, Train R2: {train_r2}")

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("validation_split", validation_split)

    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_r2", test_r2)

    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_r2", train_r2)

    model_signature = infer_signature(train_X, y_pred)

    if urlparse(URI).scheme != "file":
        mlflow.keras.log_model(
            model=model,
            artifact_path="model",
            registered_model_name="Turbofan_Engine_Life_Prediction",
            signature=model_signature,
            pip_requirements=["tensorflow==2.16.1", "keras==3.11.3"]  
        )
    else:
        mlflow.keras.log.model(
            model=model,
            artifact_path="model",
            signature=model_signature,
            registered_model_name="Turbofan_Engine_Life_Prediction".
            pip_requirements=["tensorflow==2.16.1", "keras==3.11.3"]
        )

    print("Model training completed and logged to MLflow.")


