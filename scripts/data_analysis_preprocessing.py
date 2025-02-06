"""
Data Analysis and Preprocessing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


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

    def summary_statistics(self):
        """Display summary statistics for numerical and categorical features."""
        print("Summary Statistics for numerical features:")
        print(self.data.describe())
        print("\n\n")
        print("Summary Statistics for categorical features:")
        print(self.data.describe(include=[object, 'category']))
        print("\n")

    def retrieve_numerical_columns(self):
        """Return a list of numerical columns."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def identify_missing_values(self):
        """Identify missing values in the dataset."""
        print("Missing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        print("\n")

    def outlier_detection(self):
        """Use box plots to identify outliers in numerical features."""
        print("Outlier Detection:")
        numerical_columns = self.retrieve_numerical_columns()
        for column in numerical_columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.data[column])
            plt.title(f"Boxplot of {column}")
            plt.xlabel(column)
            plt.savefig(f'../notebooks/plots/{column}_.png', dpi=300, bbox_inches='tight')
            plt.show()

    def handle_missing_values(self, strategy="mean", threshold=1.5):
        """
        Handle missing values in the dataset. Features with missing values above the threshold are dropped.
        Otherwise, missing values are handled using imputation (mean, median, or most_frequent).

        :param strategy: Imputation strategy for features with missing values below the threshold
                        ("mean", "median", "most_frequent").
        :param threshold: Proportion of missing values (between 0 and 1) to determine if a feature should be dropped.
        """
        print(f"Handling missing values using {strategy} strategy with a threshold of {threshold * 100}%...")
        initial_row_count = len(self.data)

        # Calculate the percentage of missing values for each feature
        missing_percentage = self.data.isnull().mean()

        # Identify features to drop and features to impute
        features_to_drop = missing_percentage[missing_percentage > threshold].index
        features_to_impute = missing_percentage[missing_percentage <= threshold].index

        # Drop features exceeding the threshold
        self.data = self.data.drop(columns=features_to_drop)
        print(f"Dropped features: {list(features_to_drop)}")

        # Separate features into numerical and categorical for imputation
        numerical_cols = self.data.select_dtypes(include=["number"]).columns
        categorical_cols = self.data.select_dtypes(include=["object", "category"]).columns
        print(f"Numerical columns missing values to handle on\n: {list(numerical_cols)}")
        print(f"Categorical columns missing values to handle on\n: {list(categorical_cols)}")

        # Handle imputation
        if strategy in ["mean", "median"]:
            if not numerical_cols.empty:
                imputer = SimpleImputer(strategy=strategy)
                self.data[numerical_cols] = imputer.fit_transform(self.data[numerical_cols])
                print("Handled missing values for numerical features.")
            else:
                print("No numerical columns to impute.")

        elif strategy == "most_frequent":
            if not categorical_cols.empty:
                imputer = SimpleImputer(strategy="most_frequent")
                self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
                print("Handled missing values for categorical features.")
            else:
                print("No categorical columns to impute.")

        else:
            print("Invalid strategy. Choose 'mean', 'median', or 'most_frequent'.")

        print("Missing values handled (for categorical and numerical features).")

        print(f"Missing values handled. Final row count: {len(self.data)}")
        print(f"Total rows dropped after missing value handling: {initial_row_count - len(self.data)}\n")

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        self.data.drop_duplicates(inplace=True)
        print("Duplicates removed.")

    def correct_data_types(self):
        """Convert columns to appropriate data types."""
        self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
        self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])
        print("Data types corrected.")

    def feature_engineering(self):
        """Create new features like transaction frequency and time-based features."""
        self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
        self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek
        print("Feature engineering completed.")

    def normalize_and_scale(self):
        """Normalize and scale numerical features."""
        scaler = StandardScaler()
        numerical_columns = self.retrieve_numerical_columns()
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
        print("Normalization and scaling done.")
