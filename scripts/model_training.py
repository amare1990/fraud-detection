""" Build, train, and evaluation."""

import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from scripts.smotified_gan_balancer import SMOTifiedGANBalancer
from scripts.rnn import RNNModel
from scripts.lstm import LSTMModel
from scripts.cnn import CNNModel

# Set random seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility across all necessary libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed_value}")


# Call this before anything random happen
set_seed(42)



BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection"

class FraudDetectionModel:
    """
    Params: data-processed data, target_column- target column name
    """
    def __init__(self, data, target_column):
        """
        Params: data-final balanced data to be saved,
        processed_data_path- path of processed data to get the test data
        target_column- target column name-class(0 or 1)
        """
        self.target_column = target_column
        self.balancer = SMOTifiedGANBalancer()
        self.models = {}
        self.X_train = None
        self.y_train = None

        # self.processed_data will not be processed further. Only used for extracting test data
        self.processed_data = data
        self.X_test, self.y_test = self.get_X_test_original_data()

        # This data will be processed further and finally will contain balanced data
        self.data = data

    def get_X_test_original_data(self, test_size=0.2):
        X = self.processed_data.drop(columns=[self.target_column])
        y = self.processed_data[self.target_column].values
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Original test data shape: {X_test.shape}, {y_test.shape}")
        return X_test, y_test

    def balance_data(self, target_col='class'):
        print(f"\n\n{'*'*100}\n")
        print("Balancing data using SMOTified+GAN...")

        # Features (X) and target (y) - drop the target column from X
        X = self.data.drop(columns=[target_col]).values
        y = self.data[target_col].values

        # Apply SMOTified+GAN for balancing the dataset
        X_balanced, y_balanced = self.balancer.balance_data(X, y)

        # Create a new DataFrame with the balanced data
        self.data = pd.DataFrame(X_balanced, columns=self.data.drop(columns=[target_col]).columns)
        self.data[target_col] = y_balanced

        # Save the balanced data to a CSV file
        output_path = f'{BASE_DIR}/data/balanced_processed_data.csv'
        self.data.to_csv(output_path, index=False)
        print(f"Balanced and Processed data saved to {output_path}")
        print("Data Balancing Completed Successfully.")

    def data_preparation(self, test_size=0.2):
        print(f"\n\n{'*'*120}\n")
        print("Data preparation starting...")

        # Prepare the features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column].values

        # Split the balanced data into training and test sets for model training
        self.X_train, _, self.y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Training data shape: {self.X_train.shape}, {self.y_train.shape}")
        print("Data prepared: Train and test sets created.")


    def train_sklearn_models(self):
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
        }

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

    def train_deep_learning_models(self, model_type="LSTM", epochs=10, batch_size=32, learning_rate=0.001):
        # Convert X_train and X_test to NumPy arrays before creating tensors
        if self.X_train is None or self.X_test is None:
            print("Data has not been prepared. Please call data_preparation() first.")
            return

        # Convert X_train and X_test to NumPy arrays if they are DataFrames
        # If they are already NumPy arrays, this step is skipped.
        if isinstance(self.X_train, pd.DataFrame):
            X_train_np = self.X_train.values
        else:
            X_train_np = self.X_train  # Use existing NumPy array

        if isinstance(self.X_test, pd.DataFrame):
            X_test_np = self.X_test.values
        else:
            X_test_np = self.X_test  # Use existing NumPy array

        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test_np , dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        input_size = self.X_train.shape[1]
        model = self._initialize_model(model_type, input_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(f"Training {model_type} Model...")
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
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self._plot_loss_curve(losses, model_type, epochs)
        self.models[model_type] = model

    def _initialize_model(self, model_type, input_size):
        if model_type == "CNN":
            return CNNModel(input_size)
        elif model_type == "RNN":
            return RNNModel(input_size)
        else:
            return LSTMModel(input_size)

    def _plot_loss_curve(self, losses, model_type, epochs):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='b', label=f'{model_type} Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{model_type} Training Loss Curve")
        plt.legend()
        plt.grid()
        loss_plot_path = f"{BASE_DIR}/notebooks/plots/models/{model_type}_training_loss.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved as {loss_plot_path}")
        plt.show()

    def evaluate_models(self):
        perf_result = {}

        for name, model in self.models.items():
            if isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
                                GradientBoostingClassifier, MLPClassifier)):
                y_pred = model.predict(self.X_test)
            elif isinstance(model, nn.Module):
                y_pred, true_labels = self._evaluate_model(model, self.X_test, self.y_test, name)
            else:
                continue  # Skip unsupported models

            if len(y_pred) == 0:
                print(f"Skipping {name} due to missing predictions.")
                continue  # Skip models that failed to produce predictions

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred)

            print(f"\n{name} Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nConfusion Matrix:\n", conf_matrix)
            print("\nClassification Report:\n", class_report)

            perf_result[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            # Log experiment with MLflow
            self.track_versioning_experiment(name, accuracy, params=None)

        # Ensure there's data before plotting
        if perf_result:
            self.plot_radar_chart(perf_result)
        else:
            print("No valid model metrics available, skipping radar chart.")

        return perf_result


    def _evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluates a single model and returns predictions."""
        print(f"Evaluating {model_type} model...")

        model.eval()

        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        with torch.no_grad():
            predictions = model(X_test_tensor)
            predicted_labels = (predictions >= 0.5).float().squeeze().numpy()

        return predicted_labels, y_test_tensor.numpy()


    def plot_radar_chart(self, model_metrics):
        """
        Generate a radar chart to compare model performances.
        """
        print(f"\n\n{'*'*120}\n")
        print("Generating radar chart...")
        if not model_metrics:
            print("Error: model_metrics is empty. No data to plot.")
            return

        labels = list(next(iter(model_metrics.values())).keys())
        num_vars = len(labels)

        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        for model_name, metrics in model_metrics.items():
            values = list(metrics.values())
            values += values[:1]  # Complete the loop
            ax.plot(angles, values, label=model_name, linewidth=2)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title(f'Model Comparison')
        print("\n\n")
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.savefig(f"{BASE_DIR}/notebooks/plots/models/radar_chart.png")
        plt.show()



    def save_model(self, model_name):
        print(f"\n\n{'*'*120}")
        print(f"Saving {model_name} model...")
        model = self.models.get(model_name)
        if not model:
            print(f'Model {model_name} not found!')
            return

        if model_name in ['CNN', 'RNN', 'LSTM']:
            # Define the model signature instead of input_example
            input_schema = Schema([TensorSpec(np.dtype(np.float32), shape=(-1, *self.X_test.shape[1:]))])
            output_schema = Schema([TensorSpec(np.dtype(np.float32), shape=(-1,))])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Log model with signature instead of input_example
            mlflow.pytorch.log_model(model, model_name, signature=signature)

            # Just saving with .path file extension
            torch.save(model.state_dict(), f"{BASE_DIR}/models/{model_name}.pth")

        else:
            # Save the model as a pickle file
            model_path = f"{BASE_DIR}/models/{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        print(f'Model {model_name} saved successfully!')


    def track_versioning_experiment(self, model_name, accuracy, params=None):
        print(f"\n\n{'*'*120}")
        print(f"Tracking {model_name} model")

        # Ensure no active runs before starting a new one
        if mlflow.active_run():
            mlflow.end_run()

        # Set the experiment
        mlflow.set_experiment("Fraud Detection")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_metric("accuracy", accuracy)  # Log accuracy using MLflow

            if model_name in ['CNN', 'RNN', 'LSTM']:
                # Define input schema instead of input_example
                input_schema = Schema([TensorSpec(np.dtype(np.float32), shape=(-1, *self.X_test.shape[1:]))])
                output_schema = Schema([TensorSpec(np.dtype(np.float32), shape=(-1,))])
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                # Log the model
                mlflow.pytorch.log_model(self.models[model_name], model_name, signature=signature)

            else:
                mlflow.sklearn.log_model(self.models[model_name], model_name, input_example=self.X_test[:1])

        print(f"Experiment tracked for {model_name}")
