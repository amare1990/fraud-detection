"""

Model building building, training, evalauation.
"""
import pickle

# import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score \
    , classification_report, confusion_matrix

import mlflow
import mlflow.sklearn
import mlflow.pytorch

from scripts.cnn import CNNModel
from scripts.lstm import LSTMModel


class FraudDetectionModel:
    """A machine learning model using PyTorch."""
    def __init__(self, data, target_column):
        """Initialize the class with data, target_column."""
        self.data = data
        self.target_column = target_column
        self.models = {}

        # Initialize dataset attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def data_preparation(self, test_size=0.2):
        """Prepare data for training."""
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        print("Data prepared: Train and test sets created.")

    def train_sklearn_models(self):
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
        }

        perf_result = {}

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(
                self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred)

            print(f"{name} Accuracy: {accuracy:.4f}")
            print(class_report)

            perf_result[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'conf_matrix': conf_matrix,
                'class_report': class_report
            }

        return perf_result

    def train_deep_learning_models(
            self, model_type="LSTM", epochs=10, batch_size=32, learning_rate=0.001):
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(
            self.y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(
            self.y_test, dtype=torch.float32).unsqueeze(1)

        # Reshape
        # The issue was with the reshaping for LSTM. It should be (batch_size, sequence_length, input_size)
        # For LSTM, sequence_length can be 1 if you're treating each data point as a single time step.

        # Original code:
        # X_train_tensor = X_train_tensor.unsqueeze(
        #    1) if model_type == 'CNN' else X_train_tensor.unsqueeze(2)
        # X_test_tensor = X_test_tensor.unsqueeze(
        #    1) if model_type == 'CNN' else X_test_tensor.unsqueeze(2)

        # Corrected code:
        if model_type == 'CNN':
            X_train_tensor = X_train_tensor.unsqueeze(1)
            X_test_tensor = X_test_tensor.unsqueeze(1)
        else:  # model_type == 'LSTM'
            # Assuming sequence length of 1
            X_train_tensor = X_train_tensor.unsqueeze(1)
            X_test_tensor = X_test_tensor.unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True)

        input_size = self.X_train.shape[1]
        model = CNNModel(
            input_size) if model_type == 'CNN' else LSTMModel(input_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(f'Training {model_type} Model...')
        # Store losses for visualization
        losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)  # Store loss for plotting

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Plot loss curve
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(
                1,
                epochs + 1),
            losses,
            marker='o',
            linestyle='-',
            color='b',
            label=f'{model_type} Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{model_type} Training Loss Curve")
        plt.legend()
        plt.grid()

        # Save the loss plot
        loss_plot_path = f"../notebooks/plots/{model_type}_training_loss.png"
        plt.savefig(loss_plot_path)
        print(f"Loss plot saved as {loss_plot_path}")

        # Show the plot (optional)
        plt.show()

        # Model evaluation for deep learning
        model.eval()
        correct, total = 0, 0
        predictions_list = []
        true_labels_list = []

        # Create a dictionary to store performance metrics
        perf_result = {}

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                # Assuming binary classification
                predictions = (outputs > 0.5).float()
                predictions_list.extend(
                    predictions.cpu().numpy())  # Store predictions
                true_labels_list.extend(
                    batch_y.cpu().numpy())  # Store true labels
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total

        # Convert lists to arrays for metric calculations
        predictions_array = np.array(predictions_list)
        true_labels_array = np.array(true_labels_list)

        # Calculate performance metrics
        precision = precision_score(
            true_labels_array,
            predictions_array,
            average='weighted')
        recall = recall_score(
            true_labels_array,
            predictions_array,
            average='weighted')
        f1 = f1_score(true_labels_array, predictions_array, average='weighted')
        conf_matrix = confusion_matrix(true_labels_array, predictions_array)
        class_report = classification_report(
            true_labels_array, predictions_array)

        # Print metrics
        print(f"{model_type} Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        # Store the results in perf_result
        perf_result[model_type] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'conf_matrix': conf_matrix,
            'class_report': class_report
        }

        # Save the model
        self.models[model_type] = model

        return perf_result

    def save_model(self, model_name):
        model = self.models.get(model_name)
        if not model:
            print(f'Model {model_name} not found!')
            return

        if model_name in ['CNN', 'LSTM']:
            torch.save(model.state_dict(), f'../models/{model_name}.pth')
        else:
            with open(f'../models/{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f'../models/{model_name}.pkl')

        print(f'Model {model_name} saved successfuly!')

    def track_versioning_experiment(self, model_name, accuracy, params=None):
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
            if model_name in ['CNN', 'LSTM']:
                mlflow.pytorch.log_model(self.models[model_name], model_name)
            else:
                mlflow.sklearn.log_model(self.models[model_name], model_name)

            print(f"Experiment tracked for {model_name}")
