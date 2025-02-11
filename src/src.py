"""Pipelining all processes and run automatically"""

import os
import sys


# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
sys.path.append(ROOT_DIR)
print(f'Root direc: {ROOT_DIR}')

from scripts.pipeline_model_building_processes import pipeline_model_training_processes
from scripts.data_analysis_preprocessing import FraudDataProcessor

from scripts.model_explainer import pipeline_model_explainability


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # Pipelining all data cleaning and preprocessing processes.
    fraud_detector = FraudDataProcessor()
    fraud_detector.analysis_preprocess()

    # Run all model building, training and evaluation processes automatically.

    # Pipelining all model building, training and evaluation processes.
    # Run the pipleiner
    pipeline_model_training_processes()

    # Run the model explainability pipeliner
    # Call the method
    pipeline_model_explainability()
