""" Step 5: Test the API (test_api.py)"""

import requests
import pandas as pd

# Define the absolute path to the processed data file
file_path = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data/processed_data.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Define columns to exclude
excluded_columns = ['device_id', 'signup_time', 'purchase_time', 'ip_int',
                    'lower_bound_ip_address', 'upper_bound_ip_address',
                    'lower_bound_ip_int', 'upper_bound_ip_int']

# Filter the DataFrame by dropping the excluded columns
df_filtered = df.drop(columns=excluded_columns)

# Filter out the target variable (i.e., class)
target_column = 'class'
df_filtered = df_filtered.drop(columns=[target_column])

# Select a sample (e.g., first row or a random one) for testing the API
sample_data = df_filtered.iloc[0].to_dict()  # You can change 0 to any index or use random.choice(df_filtered.index)

# Print sample data for verification
print("Sample data for API request:", sample_data)

# Make a request to the API
response = requests.post("http://127.0.0.1:5000/predict", json=sample_data)

# Print response
print(response.json())

