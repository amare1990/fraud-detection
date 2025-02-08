"""

Model building building, training, evalauation.
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



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

