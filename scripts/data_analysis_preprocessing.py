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
