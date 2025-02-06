"""
Data Analysis and Preprocessing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FraudDataProcessor:
    def __init__(self, data_path='../data/Fraud_Data.csv'):
        """
        Initialize the class with the dataset.
        :param data: Pandas DataFrame
        """
        self.data = data_path

    def overview_of_data(self):
        """Provide an overview of the dataset."""
        print("Overview of the Data:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nFirst 5 Rows:")
        print(self.data.head())
