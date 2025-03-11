"""
Pipelining model explainability.
Try Decision Tree as others take too long time to run.
"""

import pickle
import pandas as pd
import torch  # Required for loading PyTorch models

# Import DL models and their wrapper class
from scripts.cnn import CNNModel
from scripts.lstm import LSTMModel
from scripts.rnn import RNNModel
from scripts.dl_wrapper import ModelWrapper

from scripts.model_explainability import ModelExplainability


def pipeline_model_explainability():
    """A method to pipleine all processes in model explanability."""

    # Define models to load
    # Change the base directory to your preferred to one.
    base_dir_models = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/models"
    model_paths = {
        "Logistic Regression": f"{base_dir_models}/Logistic Regression.pkl",
        "Decision Tree": f"{base_dir_models}/Decision Tree.pkl",
        "Random Forest": f"{base_dir_models}/Random Forest.pkl",
        "Gradient Boosting": f"{base_dir_models}/Gradient Boosting.pkl",
        "MLP": f"{base_dir_models}/MLP Classifier.pkl",
        "CNN": f"{base_dir_models}/CNN.pth",
        "LSTM": f"{base_dir_models}/LSTM.pth",
        "RNN": f"{base_dir_models}/RNN.pth"
    }

    # Load dataset
    df = pd.read_csv(
        "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/processed_data.csv")
    target_column = "class"


    # Prepare training and testing data
    X = df.drop(columns=[target_column])
    y = df[target_column]  # Of course, this data is not used here.
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

            # You'll need to determine input_size
            input_size = X_train.shape[1]

            if model_name == 'CNN':
                model = CNNModel(input_size)
            elif model_name == 'LSTM':
                model = LSTMModel(input_size)
            elif model_name == 'RNN':
                model = RNNModel(input_size)

            state_dict = torch.load(path)  # Load state_dict
            model.load_state_dict(state_dict)  # Load state_dict into the model
            # Wrap the model to ensure predict_proba() exists
            model = ModelWrapper(model)
        else:
            print(f"Skipping unknown model format: {path}")
            continue

        base_dir = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud_detection/notebooks"

        # Initialize explainability
        explainer = ModelExplainability(
            model, X_train, X_test, feature_names, base_dir=base_dir)

        # SHAP Analysis
        print(f"Running SHAP for {model_name}...")
        explainer.shap_explain()

        # LIME Analysis
        print(f"Running LIME for {model_name}...")
        explainer.lime_explain(sample_index=5)
