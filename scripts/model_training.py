"""
Model building building, training, evalauation.
"""
import os
import pickle
import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score \
    , classification_report, confusion_matrix

import mlflow
import mlflow.sklearn
import mlflow.pytorch

from scripts.cnn import CNNModel
from scripts.rnn import RNNModel
from scripts.lstm import LSTMModel

from scripts.smotified_gan_balancer import SMOTifiedGANBalancer


# Set random seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility across all necessary libraries."""
    random.seed(seed_value)  # Python random
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch (CPU)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # PyTorch CUDA
        torch.cuda.manual_seed_all(seed_value)  # Multi-GPU
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False  # Turn off optimizations that introduce randomness

    print(f"Random seed set to: {seed_value}")

set_seed(42)  # Call this before anything random happen


class FraudDetectionModel:
    """A machine learning model using PyTorch."""
    def __init__(self, data, target_column):
        """Initialize the class with data, target_column."""
        excluded_columns = ['device_id', 'signup_time', 'purchase_time', 'ip_int',
                            'lower_bound_ip_address', 'upper_bound_ip_address',
                            'lower_bound_ip_int', 'upper_bound_ip_int']

        self.data = data.drop(columns=excluded_columns, errors='ignore')

        self.target_column = target_column

        self.balancer = SMOTifiedGANBalancer()

        self.models = {}

        # Initialize dataset attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def save_filtered_processed_data(self, output_path='/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/filtered_processed_data.csv'):
        """Save the filtered and processed data to a CSV file."""
        print("\n\n*****************************************************\n")
        if os.path.exists(output_path):
            os.remove(output_path)
        self.data.to_csv(output_path, index=False)

    def retrieve_numerical_columns(self):
        """Return a list of numerical columns."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def balance_data(self, target_col='class'):
        """
        Calls the SMOTified+GAN method to balance data.
        """
        print("Balancing data using SMOTified+GAN...")

        numerical_cols = self.retrieve_numerical_columns()
        X = self.data[numerical_cols].values
        y = self.data[target_col].values

        X_balanced, y_balanced = self.balancer.balance_data(X, y)

        # Store the balanced data
        self.data = pd.DataFrame(X_balanced, columns=numerical_cols)
        self.data[target_col] = y_balanced

        # Save the balanced data
        output_path='/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/balanced_data.csv'
        self.data.to_csv(output_path, index=False)
        print(f"Balanced and Processed data saved to {output_path}")

        print("Data Balancing Completed Successfully.")

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
            # "Support Vector Machine": SVC(),  # Added Support Vector Machine
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

            # Print performance metrics
            print(f"\n{name} Accuracy: {accuracy:.4f}")
            print(f"\n{name} Precision: {precision:.4f}")
            print(f"\n{name} Recall: {recall:.4f}")
            print(f"\n{name} F1-Score: {f1:.4f}")

            print(f"\n{name} Confusion Matrix:")
            print("-" * 40)
            print(conf_matrix)

            print("-" * 40)
            print(f"\n{name} class_report:")
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


        # Reshape data to add sequence_length dimension
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

        # Initialize model based on model_type
        if model_type == 'CNN':
            model = CNNModel(input_size)
        elif model_type == 'RNN':
            model = RNNModel(input_size)  # Initialize RNNModel
        else:  # model_type == 'LSTM'
            model = LSTMModel(input_size)

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
        loss_plot_path = f"/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/notebooks/plots/{model_type}_training_loss.png"
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
        print(f"\n{model_type} Accuracy: {accuracy:.4f}")
        print(f"\n {model_type} Precision: {precision:.4f}")
        print(f"\n {model_type} Recall: {recall:.4f}")
        print(f"\n {model_type} F1-Score: {f1:.4f}")

        # Print confusion matrix
        print(f"\n{model_type} - Confusion Matrix:")
        print("-" * 40)
        print(conf_matrix)

        # Print classification report
        print(f"\n{model_type} - Classification Report:")
        print("-" * 40)
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

        if model_name in ['CNN', 'RNN', 'LSTM']:
            torch.save(model.state_dict(), f"/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/models/{model_name}.pth")
        else:
            # Open the file in write binary mode ('wb')
            with open(f"/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/models/{model_name}.pkl", 'wb') as f:
                pickle.dump(model, f)  # Pass the file object 'f' to pickle.dump

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
            if model_name in ['CNN', 'RNN', 'LSTM']:
                mlflow.pytorch.log_model(self.models[model_name], model_name)
            else:
                mlflow.sklearn.log_model(self.models[model_name], model_name)

            print(f"Experiment tracked for {model_name}")
