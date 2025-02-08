"""

Model building building, training, evalauation.
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix



class FraudDetectionModel:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.models = {}


    def data_preparation(self, test_size=0.2):
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
            precision = precision_score(self.y_test, y_pred, average='weighted')
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




