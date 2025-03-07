"""Peipeline all model building processes."""

import pandas as pd

from scripts.model_training import FraudDetectionModel

df = pd.read_csv(
    "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/processed_data.csv")
target_column = 'class'
train_pipeliner = FraudDetectionModel(df, target_column)


def pipeline_model_training_processes():
    """Run all model building, training and evaluation processes methods."""
    train_pipeliner.balance_data()
    train_pipeliner.data_preparation()
    train_pipeliner.train_sklearn_models()

    train_pipeliner.train_deep_learning_models(
        model_type="RNN", epochs=10, batch_size=32)
    train_pipeliner.train_deep_learning_models(
        model_type="LSTM", epochs=10, batch_size=32)
    train_pipeliner.train_deep_learning_models(
        model_type="CNN", epochs=10, batch_size=32)

    # Evaluate models
    performance_results = train_pipeliner.evaluate_models()

    train_pipeliner.save_model("Logistic Regression")
    train_pipeliner.save_model("Decision Tree")
    train_pipeliner.save_model("Random Forest")
    train_pipeliner.save_model("Gradient Boosting")
    train_pipeliner.save_model("MLP Classifier")

    train_pipeliner.save_model("RNN")
    train_pipeliner.save_model("LSTM")
    train_pipeliner.save_model("CNN")

    # Print the evaluation results
    for model_name, metrics in performance_results.items():
        print(f"\nPerformance for {model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Confusion Matrix:\n{metrics['conf_matrix']}")
        print(f"Classification Report:\n{metrics['class_report']}")
