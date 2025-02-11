"""Pipeliner for model explanability. """

import pickle
import pandas as pd
import torch  # Required for loading PyTorch models
# Ensure your explainability class is imported
from scripts.model_explainability import ModelExplainability


def pipeline_model_explainability():
    """A method to pipleine all processes in model explanability."""
    # Define models to load
    models_dir = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/models"
    model_paths = {
        "Logistic Regression": f"{models_dir}/Logistic Regression.pkl",
        "Decision Tree": f"{models_dir}/Decision Tree.pkl",
        "Random Forest": f"{models_dir}/Random Forest.pkl",
        "Gradient Boosting": f"{models_dir}Gradient Boosting.pkl",
        "MLP": f"{models_dir}/MLP.pkl",
        "CNN": f"{models_dir}/CNN.pth",
        "LSTM": f"{models_dir}/LSTM.pth",
        "RNN": f"{models_dir}/RNN.pth"
    }

    # Load dataset
    df = pd.read_csv(
        "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/processed_data.csv")
    target_column = "class"

    # Drop unwanted columns
    excluded_columns = [
        'device_id', 'signup_time', 'purchase_time', 'ip_int',
        'lower_bound_ip_address', 'upper_bound_ip_address',
        'lower_bound_ip_int', 'upper_bound_ip_int'
    ]
    df = df.drop(columns=excluded_columns, errors='ignore')

    # Prepare training and testing data
    X = df.drop(columns=[target_column])
    # y = df[target_column]
    X_train, X_test = X.iloc[:int(0.8 * len(X))], X.iloc[int(0.8 * len(X)):]
    # Ensures feature names match the dataset
    feature_names = X_train.columns.tolist()

    # Loop through each model and run explainability
    for model_name, path in model_paths.items():
        print(f"\nLoading model: {model_name}...")

        # Load model
        if path.endswith(".pkl"):  # Machine Learning models
            with open(path, 'rb') as f:
                model = pickle.load(f)
        elif path.endswith(".pth"):  # Deep Learning models (PyTorch)
            model = torch.load(path)  # Load the model state_dict
            model.eval()  # Set to evaluation mode for inference
        else:
            print(f"Skipping unknown model format: {path}")
            continue

        base_dir = "home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/notebooks"

        # Initialize explainability
        explainer = ModelExplainability(
            model, X_train, X_test, feature_names, base_dir=base_dir)

        # SHAP Analysis
        print(f"Running SHAP for {model_name}...")
        explainer.shap_explain()

        # LIME Analysis
        print(f"Running LIME for {model_name}...")
        explainer.lime_explain(sample_index=5)
