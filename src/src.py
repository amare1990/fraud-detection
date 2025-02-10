"""Pipelining all processes and run automatically"""

import os
import sys

import pandas as pd


# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
sys.path.append(ROOT_DIR)
print(f'Root direc: {ROOT_DIR}')

from scripts.model_training import FraudDetectionModel
from scripts.data_analysis_preprocessing import FraudDataProcessor


if __name__ == '__main__':

    # Pipelining all data cleaning and preprocessing processes.
    fraud_detector = FraudDataProcessor()
    fraud_detector.analysis_preprocess()

    # Run all model building, training and evaluation processes automatically.
    # Load data (in pandas) and assign the 'class' variable to the
    # target_column variable.
    df = pd.read_csv(
        '/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/processed_data.csv')
    target_column = "class"

    """Pipelining all model building, training and evaluation processes."""
    fraud_detector = FraudDetectionModel(df, target_column)

    fraud_detector.pipeline_model_training_processes()
