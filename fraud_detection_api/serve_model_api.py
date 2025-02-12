"""Create the Flask API (serve_model.py)"""

import os, sys

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging
import torch  # Required for loading PyTorch models


# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.cnn import CNNModel  # Ensure this file exists
from scripts.lstm import LSTMModel  # Ensure this file exists
from scripts.rnn import RNNModel  # Ensure this file exists

from scripts.model_training import FraudDetectionModel

# Initialize Flask app
app = Flask(__name__)

# Configure Logging
logging.basicConfig(filename='fraud_api.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load the trained models
models_path = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/models"
model_paths = {
    "Logistic Regression": f"{models_path}/Logistic Regression.pkl",
    "Decision Tree": f"{models_path}/Decision Tree.pkl",
    "Random Forest": f"{models_path}/Random Forest.pkl",
    "Gradient Boosting": f"{models_path}/Gradient Boosting.pkl",
    "MLP Classifier": f"{models_path}/MLP Classifier.pkl",
    "CNN": f"{models_path}/CNN.pth",
    "LSTM": f"{models_path}/LSTM.pth",
    "RNN": f"{models_path}/RNN.pth"
}

df = pd.read_csv("/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/processed_data.csv")
target_column = 'class'

train_pipeliner = FraudDetectionModel(df, target_column)
train_pipeliner.data_preparation()
X = train_pipeliner.data.drop(columns=[train_pipeliner.target_column])


input_size = X.shape[1]

# train_pipeliner = FraudDetectionModel(df, target_column)

# Dictionary to store loaded models
models = {}

# Load all models
for model_name, path in model_paths.items():
    print(f"\nLoading model: {model_name}...")

    try:
        if path.endswith(".pkl"):  # Machine Learning models
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)

        elif path.endswith(".pth"):  # Deep Learning models (PyTorch)

            if model_name == "CNN":
                model = CNNModel(input_size=input_size)  # Set your correct input size
            elif model_name == "LSTM":
                model = LSTMModel(input_size=input_size)
            elif model_name == "RNN":
                model = RNNModel(input_size=input_size)

            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
            models[model_name] = model

        print(f"Loaded: {model_name}")

    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Ensure the model name is provided in the request
        model_name = data.get("model_name", "Logistic Regression")  # Default to Logistic Regression

        if model_name not in models:
            return jsonify({"error": f"Model '{model_name}' not found"}), 400

        model = models[model_name]

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Make predictions
        prediction = model.predict(df)
        probability = model.predict_proba(df)

        result = {
            "model_used": model_name,
            "prediction": int(prediction[0]),
            "fraud_probability": float(probability[0][1])
        }

        # Log request and response
        logging.info(f"Received: {data} | Prediction: {result}")

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
