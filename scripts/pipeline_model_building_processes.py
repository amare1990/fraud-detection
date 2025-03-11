"""
Peipeline all model building processes.
The dataset should be the processed data.
"""

import pandas as pd

from scripts.model_training import FraudDetectionModel

BASE_DIR = '/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection'

# For bank credit card transaction data, 'creditcard.csv'
df = pd.read_csv(
    f"{BASE_DIR}/data/processed_data_Bank transaction.csv")

# For e-Commerce transaction data, 'Fraud_Data.csv'
df = pd.read_csv(
    f"{BASE_DIR}/data/processed_data_e-Commerce transaction.csv")


# For e-Commerce transaction data, the target variable is 'class'.
# If it is a bank transaction data, the target variable is 'Class' rather.
target_column = 'class'

# If load balancing is applied, the class instantiation creates test data.
# The test data is from the imbalanced data
train_pipeliner = FraudDetectionModel(df, target_column)


def pipeline_model_training_processes():
    """Run all model building, training and evaluation processes methods."""
    train_pipeliner.balance_data()

    # Get the training data from the balanced data
    train_pipeliner.data_preparation()

    # Train traditional ML models including MLP
    train_pipeliner.train_sklearn_models()

    # Train DL models
    train_pipeliner.train_deep_learning_models(
        model_type="RNN", epochs=50, batch_size=32)
    train_pipeliner.train_deep_learning_models(
        model_type="LSTM", epochs=50, batch_size=32)
    train_pipeliner.train_deep_learning_models(
        model_type="CNN", epochs=50, batch_size=32)

    # Evaluate models
    performance_results = train_pipeliner.evaluate_models()

    # Save models for further usage
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
