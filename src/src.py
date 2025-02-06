"""Pipelining all processes and to run automatically"""

from scripts.data_analysis_preprocessing import FraudDataProcessor



if __name__ == '__src__':
  """Pipelining all data cleaning and preprocessing processes."""
  fraud_detector = FraudDataProcessor()
  fraud_detector.analysis_preprocess()
