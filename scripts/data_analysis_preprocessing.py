"""
Data Analysis and Preprocessing
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

from scipy.stats import chi2_contingency


BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection"


class FraudDataProcessor:
    def __init__(
            self,
            data_path=f'{BASE_DIR}/data/Fraud_Data.csv'):
        """
        Initialize the class with the dataset.
        :param data: Pandas DataFrame
        """
        self.data = pd.read_csv(data_path)

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
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/outliers/outlier_{column}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

    def handle_missing_values(self, strategy="mean", threshold=0.3):
        """
        Handle missing values in the dataset. Features with missing values above the threshold are dropped.
        Otherwise, missing values are handled using imputation (mean, median, or most_frequent).
        """
        print("\n\n*****************************************************\n")
        print(
            f"Handling missing values using {strategy} strategy with a threshold of {threshold * 100}%...")

        initial_row_count = len(self.data)

        # Calculate missing value percentages
        missing_percentage = self.data.isnull().mean()

        # Drop columns exceeding the threshold
        features_to_drop = missing_percentage[missing_percentage >
                                              threshold].index
        self.data = self.data.drop(columns=features_to_drop)
        print(f"Dropped features: {list(features_to_drop)}")

        # Separate numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=["number"]).columns
        categorical_cols = self.data.select_dtypes(
            include=["object", "category"]).columns

        print(
            f"Numerical columns missing values to handle on: {list(numerical_cols)}")
        print(
            f"Categorical columns missing values to handle on: {list(categorical_cols)}")

        # Convert categorical columns to string before imputation
        self.data[categorical_cols] = self.data[categorical_cols].astype(str)

        # Impute numerical columns
        if strategy in ["mean", "median"]:
            if not numerical_cols.empty:
                imputer = SimpleImputer(strategy=strategy)
                self.data[numerical_cols] = imputer.fit_transform(
                    self.data[numerical_cols])

        # Impute categorical columns with most frequent values
        if not categorical_cols.empty:
            imputer = SimpleImputer(strategy="most_frequent")
            self.data[categorical_cols] = imputer.fit_transform(
                self.data[categorical_cols])

        # Drop one-hot encoded `_nan` columns if they exist
        self.data = self.data.drop(
            columns=[
                col for col in self.data.columns if "_nan" in col],
            errors="ignore")

        # Drop any remaining NaN rows (if any)
        self.data = self.data.dropna()

        print(f"Missing values handled. Final row count: {len(self.data)}")
        print(f"Total rows dropped: {initial_row_count - len(self.data)}")

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        self.data.drop_duplicates(inplace=True)
        print("Duplicates removed.")

    def correct_data_types(self):
        """Convert columns to appropriate data types."""
        print(
            f"Data type of signup_time before correction: {self.data['signup_time'].dtype}")
        print(
            f"Data type of purchase_time before correction: {self.data['purchase_time'].dtype}")
        self.data['signup_time'] = pd.to_datetime(self.data['signup_time'])
        self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])
        print("Data types corrected.")
        print(
            f"Data type of signup_time after correction: {self.data['signup_time'].dtype}")
        print(
            f"Data type of purchase_time after correction: {self.data['purchase_time'].dtype}")

    def univariate_analysis(self):
        """Perform univariate analysis on numerical columns."""
        print("Univariate Analysis starting: numerical columns")
        numerical_columns = self.retrieve_numerical_columns()
        for col in numerical_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Univariate Analysis - {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/univariante/numerical/hist_{col}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

        """Perform univariante analysis on categorical columns."""
        print("Univariate Analysis for categorical columns starting...")
        categorical_columns = self.data.select_dtypes(
            include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.data[col])
            plt.title(f"Univariante Analysis - {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/univariante/categorical/countplot_{col}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

    def bivariate_analysis(self):
        """Perform bivariate analysis (correlation, pair plots and box plot)."""
        # Correlation Heatmap
        print("Correlation Heatmap:")
        # Convert list to DataFrame
        numerical_columns = self.data[self.retrieve_numerical_columns()]
        plt.figure(figsize=(12, 8))
        corr_matrix = numerical_columns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(
            f'{BASE_DIR}/notebooks/plots/heatmap.png',
            dpi=300,
            bbox_inches='tight')
        plt.show()

        # Pair Plot (only a subset of columns for better visualization)
        print("\nBivariate Analysis - Pair Plot: Starting...")
        numerical_columns = self.retrieve_numerical_columns()
        # Adjust number of columns to display in pair plot
        subset = numerical_columns[:5]
        sns.pairplot(
            self.data[subset],
            diag_kind='kde',
            plot_kws={
                'alpha': 0.5})
        plt.title("Bivariate Analysis - Pair Plot")
        plt.savefig(
            f'{BASE_DIR}/notebooks/plots/bivariante/pairplot.png',
            dpi=300,
            bbox_inches='tight')
        plt.show()

        # Bivariate analysis on categorical
        print("Bivariate Analysis - Boxplot:")
        # categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_columns = ['browser', 'source', 'sex']
        for col in categorical_columns:
            # if col != 'class':  # Avoid 'class' as target variable
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[col], y=self.data['class'])
            plt.title(f"Bivariate Analysis - {col} vs class")
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/bivariante/categ/boxplot_{col}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

    # Bivariante analysis between categorical variables
    def bivariate_categorical_analysis(self):
        """Perform bivariate analysis between two categorical columns."""
        print("Bivariate Analysis: contigency table - browser vs source: starting")
        # Create a contingency table (cross-tabulation)
        contingency_table = pd.crosstab(
            self.data['browser'], self.data['source'])
        print("Contingency Table (browser vs source):")
        print(contingency_table)

        # Visualize the relationship using a stacked bar chart (countplot)
        print("Bivariate Analysis - browser vs source: countplot plotting")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='browser', hue='source', data=self.data)
        plt.title("Bivariate Analysis - browser vs source")
        plt.savefig(
            f'{BASE_DIR}/notebooks/plots/bivariante/categ_stacked_bar_chart.png',
            dpi=300,
            bbox_inches='tight')
        plt.show()

        # Optional: Chi-square test of independence
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Test p-value: {p}")
        if p < 0.05:
            print("The variables are likely dependent.")
        else:
            print("The variables are likely independent.")

    def ip_to_integer(self, ip):
        """Convert an IP address stored as float64 to an integer."""
        try:
            ip = int(ip)  # Convert float64 to int
            if 0 <= ip <= 4294967295:  # Ensure valid IPv4 range
                return ip
            else:
                print(f"Warning: Out-of-range IP encountered: {ip}")
                return None
        except (ValueError, TypeError):
            print(f"Warning: Invalid IP address encountered: {ip}")
            return None

    def merge_datasets_for_geolocation(self):
        """Merge Fraud_Data.csv with IpAddress_to_Country.csv using IP ranges."""
        # Convert IP addresses to integer format
        self.data['ip_int'] = self.data['ip_address'].apply(self.ip_to_integer)

        # Drop invalid IPs
        self.data = self.data.dropna(subset=['ip_int'])
        self.data['ip_int'] = self.data['ip_int'].astype(
            int)  # Ensure int type

        print(f"Number of valid IPs: {self.data.shape[0]}")

        # Load IP-to-country mapping
        ip_to_country = pd.read_csv(
            f'{BASE_DIR}/data/IpAddress_to_Country.csv')

        # Convert lower and upper bounds to integer format
        ip_to_country['lower_bound_ip_int'] = ip_to_country['lower_bound_ip_address'].apply(
            self.ip_to_integer)
        ip_to_country['upper_bound_ip_int'] = ip_to_country['upper_bound_ip_address'].apply(
            self.ip_to_integer)

        # Drop invalid rows
        ip_to_country = ip_to_country.dropna(
            subset=['lower_bound_ip_int', 'upper_bound_ip_int'])
        ip_to_country['lower_bound_ip_int'] = ip_to_country['lower_bound_ip_int'].astype(
            int)
        ip_to_country['upper_bound_ip_int'] = ip_to_country['upper_bound_ip_int'].astype(
            int)

        print(f"Valid IP ranges in dataset: {ip_to_country.shape[0]}")

        # Sort both datasets for merge_asof (which requires sorted data)
        self.data = self.data.sort_values(by='ip_int')
        ip_to_country = ip_to_country.sort_values(by='lower_bound_ip_int')

        # Perform a range-based merge using merge_asof
        merged_data = pd.merge_asof(
            self.data, ip_to_country,
            left_on='ip_int',
            right_on='lower_bound_ip_int',
            direction='backward'  # Ensure we find the closest lower bound
        )

        # Filter out rows where IP exceeds the upper bound
        merged_data = merged_data[merged_data['ip_int']
                                  <= merged_data['upper_bound_ip_int']]

        print(f"Final merged data shape: {merged_data.shape}")

        self.data = merged_data  # Save final merged dataset
        print("Datasets successfully merged for geolocation analysis.")

    def feature_engineering(self):
        """Create new features like hours_of_day, hours_of_week and calculate transaction frequency and time-based features."""
        print("Feature engineering starting...")

        self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
        self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek

        # Ensure timestamps are sorted correctly
        self.data = self.data.sort_values(by=['user_id', 'purchase_time'])

        # Compute transaction frequency
        self.data['transaction_frequency'] = self.data.groupby(
            'user_id')['purchase_time'].diff().dt.total_seconds()

        # Ensure non-negative values
        self.data['transaction_frequency'] = self.data['transaction_frequency'].apply(
            lambda x: max(x, 0))

        self.data['transaction_velocity'] = (
            self.data['purchase_time'] -
            self.data['signup_time']).dt.total_seconds()
        self.data['transaction_velocity'] = self.data['transaction_velocity'].apply(
            lambda x: max(x, 0))

        # If the same device_id is used by multiple user_ids, it may indicate
        # fraudulent accounts.
        self.data['device_shared_count'] = self.data.groupby(
            'device_id')['user_id'].transform('nunique')

        # Age Grouping (Binning)
        self.data['age_group'] = pd.cut(
            self.data['age'], bins=[
                0, 18, 30, 45, 60, 100], labels=[
                'Teen', 'Young', 'Adult', 'Middle_Aged', 'Senior'])

    def normalize_and_scale(self):
        """Normalize and scale numerical features."""
        print("Normalizing and scaling numerical features... starting")
        scaler = MinMaxScaler()
        scalable_columns = [
            'transaction_frequency',
            'transaction_velocity',
            'purchase_value',
            'device_shared_count']
        print(f"Scalable columns: {scalable_columns}")
        self.data[scalable_columns] = scaler.fit_transform(
            self.data[scalable_columns])
        print("Normalization and scaling done.")

    def encode_categorical_features(self):
        """Encode categorical features:
        - 'sex' using Label Encoding.
        - 'browser', 'source', 'country', 'age_group' using One-Hot Encoding.
        """
        print("\n\n*****************************************************\n")
        print("Encoding selected categorical features... starting")

        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown='ignore')

        # Define categorical feature groups
        label_encoding_column = 'sex'  # Label encode 'sex'
        onehot_encoding_columns = [
            'browser',
            'source',
            'country',
            'age_group']  # One-hot encode the rest

        # Ensure only existing columns are selected
        onehot_encoding_columns = [
            col for col in onehot_encoding_columns if col in self.data.columns]

        # Convert to string to handle missing values and prevent issues
        self.data[label_encoding_column] = self.data[label_encoding_column].astype(
            str)
        self.data[onehot_encoding_columns] = self.data[onehot_encoding_columns].astype(
            str)

        # Apply Label Encoding to 'sex'
        if label_encoding_column in self.data.columns:
            self.data[label_encoding_column] = label_encoder.fit_transform(
                self.data[label_encoding_column])

        # Apply One-Hot Encoding to the remaining categorical features
        if onehot_encoding_columns:
            encoded_data = onehot_encoder.fit_transform(
                self.data[onehot_encoding_columns])
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=[
                    f"{col}_{category}" for col,
                    categories in zip(
                        onehot_encoding_columns,
                        onehot_encoder.categories_) for category in categories]
            )

            # Drop original categorical columns and merge one-hot encoded data
            self.data = self.data.drop(columns=onehot_encoding_columns)
            self.data = pd.concat([self.data, encoded_df], axis=1)

        # Drop any unwanted "_nan" columns and remove NaNs
        self.data = self.data.drop(
            columns=[
                col for col in self.data.columns if "_nan" in col],
            errors="ignore")
        self.data = self.data.dropna()

        print("Categorical encoding completed.")

    def save_processed_data(
            self,
            output_path=f'{BASE_DIR}/data/processed_data.csv'):

        # Ensure unnecessary columns are droppped
        excluded_columns = ['user_id', 'age', 'device_id', 'signup_time', 'purchase_time', 'ip_address', 'ip_int',
                            'lower_bound_ip_address', 'upper_bound_ip_address',
                            'lower_bound_ip_int', 'upper_bound_ip_int']

        self.data = self.data.drop(columns=excluded_columns, errors='ignore')
        """Save the processed data to a CSV file."""
        self.data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    def analysis_preprocess(self):
        """Run all instance methods."""
        self.overview_of_data()
        self.summary_statistics()
        self.retrieve_numerical_columns()
        self.identify_missing_values()
        self.outlier_detection()
        self.handle_missing_values()
        self.remove_duplicates()
        self.correct_data_types()

        # Exploratory Data Analysis/ Visualizations
        self.univariate_analysis()
        self.bivariate_analysis()
        self.bivariate_categorical_analysis()

        self.merge_datasets_for_geolocation()
        self.feature_engineering()
        self.normalize_and_scale()
        self.encode_categorical_features()
        self.save_processed_data()
