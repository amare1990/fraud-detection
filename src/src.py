"""Pipelining all processes and to run automatically"""

import os, sys

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
sys.path.append(ROOT_DIR)
print(f'Root direc: {ROOT_DIR}')

from scripts.data_analysis_preprocessing import FraudDataProcessor


if __name__ == '__main__':
  """Pipelining all data cleaning and preprocessing processes."""
  fraud_detector = FraudDataProcessor()
  fraud_detector.analysis_preprocess()
