"""Tracks versioning, logging of a model."""

import mlflow

from scripts.model_training import FraudDetectionModel

df = "../data/processed_data.csv"
target_column = 'class'
train_pipeliner = FraudDetectionModel(df, target_column)



def track_versioning_experiment(model_name, accuracy, params=None):
        # Set the experiment (ideally should be done once, not every run)
        mlflow.set_experiment("Fraud Detection")

        with mlflow.start_run():
            # Log model name as a parameter
            mlflow.log_param("Model", model_name)
            # Log accuracy as a metric
            mlflow.log_metric("Accuracy", accuracy)

            # Log hyperparameters if provided
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)

            # Log model based on type (CNN or LSTM for PyTorch; else for
            # scikit-learn)
            if model_name in ['CNN', 'RNN', 'LSTM']:
                mlflow.pytorch.log_model(train_pipeliner.models[model_name], model_name)
            else:
                mlflow.sklearn.log_model(train_pipeliner.models[model_name], model_name)

            print(f"Experiment tracked for {model_name}")
