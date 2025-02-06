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
            plt.savefig(f'../notebooks/plots/outlier_{column}.png', dpi=300, bbox_inches='tight')
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

    def univariate_analysis(self):
        """Perform univariate analysis on numerical columns."""
        print("Univariate Analysis starting: numerical columns")
        numerical_columns = self.retrieve_numerical_columns()
        for col in numerical_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Univariate Analysis - {col}")
            plt.xlabel(col)  # X-label for numerical data
            plt.ylabel("Frequency")  # Y-label indicating frequency
            plt.savefig(f'../notebooks/plots/univariante/hist_{col}.png', dpi=300, bbox_inches='tight')
            plt.show()

        """Perform univariante analysis on categorical columns."""
        print("Univariate Analysis starting: categorical columns")
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            plt.figure(figsize=(10,6))
            sns.countplot(x=self.data[col])
            plt.title(f"Univariante Analysis - {col}")
            plt.xlabel(col)  # X-label indicating the categorical variable
            plt.ylabel("Count")  # Y-label indicating the count of occurrences
            plt.savefig(f'../notebooks/plots/univariante/countplot_{col}.png', dpi=300, bbox_inches='tight')
            plt.show()

    def bivariate_analysis(self):
        """Perform bivariate analysis (correlation, pair plots and box plot)."""
        # Correlation Heatmap
        print("Correlation Heatmap:")
        plt.figure(figsize=(12, 8))
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig('notebooks/plots/heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Pair Plot (only a subset of columns for better visualization)
        print("Bivariate Analysis - Pair Plot:")
        numerical_columns = self.retrieve_numerical_columns()
        subset = numerical_columns[:5]  # Adjust number of columns to display in pair plot
        sns.pairplot(self.data[subset], diag_kind='kde', plot_kws={'alpha': 0.5})
        plt.title("Bivariate Analysis - Pair Plot")
        plt.savefig('notebooks/plots/bivariante/paiplot.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Bivariate analysis on categorical vs numerical (Boxplot example)
        print("Bivariate Analysis - Boxplot:")
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            if col != 'class':  # Avoid 'class' as target variable
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.data[col], y=self.data['purchase_value'])
                plt.title(f"Bivariate Analysis - {col} vs purchase_value")
                plt.savefig(f'notebooks/plots/bivariante/boxplot_{col}.png', dpi=300, bbox_inches='tight')
                plt.show()

    # Bivariante analysis between categorical variables
    def bivariate_categorical_analysis(self):
        """Perform bivariate analysis between two categorical columns."""
        print("Bivariate Analysis: contigency table - browser vs source: starting")
        # Create a contingency table (cross-tabulation)
        contingency_table = pd.crosstab(self.data['browser'], self.data['source'])
        print("Contingency Table (browser vs source):")
        print(contingency_table)

        # Visualize the relationship using a stacked bar chart (countplot)
        print("Bivariate Analysis - browser vs source: countplot plotting")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='browser', hue='source', data=self.data)
        plt.title("Bivariate Analysis - browser vs source")
        plt.savefig(f'notebooks/plots/bivariante/categ_stacked_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Optional: Chi-square test of independence
        from scipy.stats import chi2_contingency
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Test p-value: {p}")
        if p < 0.05:
            print("The variables are likely dependent.")
        else:
            print("The variables are likely independent.")


    def feature_engineering(self):
        """Create new features like hours_of_day, hours_of_week and calculate transaction frequency and time-based features."""
        print("Feature engineering starting...")

        self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
        self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek

        # Transaction Frequency: Count the number of transactions per user within a 24-hour period.
        self.data['transaction_frequency'] = self.data.groupby('user_id')['purchase_time'].transform(
        lambda x: x.diff().dt.total_seconds().le(86400).cumsum())

        # Transaction Velocity: Calculate the total amount spent by each user in the last 24 hours.
        self.data['transaction_velocity'] = self.data.groupby('user_id')['purchase_value'].transform(
        lambda x: x.rolling('1D', on='purchase_time').sum())
        print("Feature engineering completed.")

    def normalize_and_scale(self):
        """Normalize and scale numerical features."""
        print("Normalizing and scaling numerical features... starting")
        scaler = StandardScaler()
        numerical_columns = self.retrieve_numerical_columns()
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])
        print("Normalization and scaling done.")

    def encode_categorical_features(self, method ='onehot'):
        """Encode categorical features using Label Encoding."""
        print("Encoding categorical features... starting")
        label_encoder = LabelEncoder()
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if method == 'onehot':

            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

            for col in categorical_columns:
              # Reshape and fit transform
              encoded_data = encoder.fit_transform(self.data[[col]])
              encoded_df = pd.DataFrame(encoded_data, columns=[f"{col}_{category}" for category in encoder.categories_[0]])

              # Drop the original column and concatenate the one-hot encoded columns
              self.data = self.data.drop(col, axis=1)
              self.data = pd.concat([self.data, encoded_df], axis=1)
        else:
            for col in categorical_columns:
              self.data[col] = label_encoder.fit_transform(self.data[col])
        print(f"Categorical encoding using {method} completed.")

    def save_processed_data(self, output_path='../data/processed_data.csv'):
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
        self.univariate_analysis()
        self.bivariate_analysis()
        self.bivariate_categorical_analysis()
        self.feature_engineering()
        self.normalize_and_scale()
        self.encode_categorical_features()
        self.save_processed_data()
