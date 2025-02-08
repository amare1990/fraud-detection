"""

Model building building, training, evalauation.
"""
import pandas as pd
import numpy as np




class FraudDetectionModel:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.models = {}
