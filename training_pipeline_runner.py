from model_training_pipeline import ModelTrainingPipeline
from src.test_model import test_model
import mlflow
import sys


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
    model = trainer.train_model()

    train_X, train_y, test_X, test_y = trainer.load_data()
    y_pred = model.predict(test_X).reshape(1, -1)[0]

    test_loss, test_mae, test_accuracy_score = test_model(model, test_X, test_y)
    train_loss, train_mae, train_accuracy_score = test_model(model, train_X, train_y)

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("validation_split", validation_split)

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_accuracy", test_accuracy_score)

    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_accuracy", train_accuracy_score)

    mlflow.keras.log_model(model, "model")
    print("Model training completed and logged to MLflow.")