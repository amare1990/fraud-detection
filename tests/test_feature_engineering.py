import pandas as pd
import pytest
from scripts.data_analysis_preprocessing import FraudDataProcessor


def test_device_shared_count():
    # Sample test data
    test_data = pd.DataFrame({
        'device_id': ['D1', 'D1', 'D2', 'D3', 'D3', 'D3'],
        'user_id': ['U1', 'U2', 'U3', 'U4', 'U5', 'U4']
    })

    preprocessor = FraudDataProcessor(test_data)

    # Apply feature engineering method
    preprocessor.data['device_shared_count'] = preprocessor.data.groupby(
        'device_id')['user_id'].transform('nunique')

    # Expected result: Unique user count per device_id
    expected_counts = [2, 2, 1, 2, 2, 2]  # D1 has 2 users, D2 has 1, D3 has 2

    assert preprocessor.data['device_shared_count'].tolist(
    ) == expected_counts, "Device shared count calculation is incorrect"
