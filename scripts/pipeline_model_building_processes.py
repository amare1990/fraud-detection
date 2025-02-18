"""Peipeline all model building processes."""

import pandas as pd

from scripts.model_training import FraudDetectionModel

df = pd.read_csv("/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/processed_data.csv")
target_column = 'class'
train_pipeliner = FraudDetectionModel(df, target_column)


def pipeline_model_training_processes():
    """Run all model building, training and evaluation processes methods."""
    # trainer = FraudDetectionModel(df, target_column)

    train_pipeliner.save_filtered_processed_data()
    train_pipeliner.balance_data()
    train_pipeliner.data_preparation()
    train_pipeliner.train_sklearn_models()

    train_pipeliner.train_deep_learning_models(model_type="RNN", epochs=10)
    train_pipeliner.train_deep_learning_models(model_type="LSTM", epochs=10)
    train_pipeliner.train_deep_learning_models(model_type="CNN", epochs=10)

    train_pipeliner.save_model("Logistic Regression")
    # trainer.save_model("Support Vector Machine")
    train_pipeliner.save_model("Decision Tree")
    train_pipeliner.save_model("Random Forest")
    train_pipeliner.save_model("Gradient Boosting")
    train_pipeliner.save_model("MLP Classifier")

    train_pipeliner.save_model("RNN")
    train_pipeliner.save_model("LSTM")
    train_pipeliner.save_model("CNN")

    train_pipeliner.track_versioning_experiment(
        "Logistic Regression", accuracy=0.98)
    # trainer.track_versioning_experiment("Support Vector Machine", accuracy=0.98)
    train_pipeliner.track_versioning_experiment("Decision Tree", accuracy=0.98)
    train_pipeliner.track_versioning_experiment("Random Forest", accuracy=0.98)
    train_pipeliner.track_versioning_experiment(
        "Gradient Boosting", accuracy=0.98)
    train_pipeliner.track_versioning_experiment(
        "MLP Classifier", accuracy=0.98)
    train_pipeliner.track_versioning_experiment("RNN", accuracy=0.98)
    train_pipeliner.track_versioning_experiment("LSTM", accuracy=0.98)
    train_pipeliner.track_versioning_experiment("CNN", accuracy=0.98)
