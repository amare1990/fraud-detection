"""
Data Analysis and Preprocessing
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

from scipy.stats import chi2_contingency


BASE_DIR = "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection"


class FraudDataProcessor:
    def __init__(self, data_path=None):
        """
        Initialize the class with the dataset.
        :param data_path: Path to the dataset CSV file.
        """
        # Set default path if None is provided. Only just give 'Fraud_Data.csv'
        if data_path is None:
            data_path = f'{BASE_DIR}/data/Fraud_Data.csv'

        self.data_path = data_path

        # Check the dataset type and read the data
        if 'Fraud_Data.csv' in self.data_path:
            self.data = pd.read_csv(self.data_path)
            self.transaction_type = 'e-Commerce transaction'
        elif 'creditcard.csv' in self.data_path:
            self.data = pd.read_csv(self.data_path)
            self.transaction_type = 'Bank transaction'
        elif 'IpAddress_to_Country.csv' in self.data_path:
            self.data = pd.read_csv(self.data_path)
            self.transaction_type = 'Geolocation'
        else:
            raise ValueError("Unknown dataset. Please provide a valid data file.")


    def overview_of_data(self):
        """Provide an overview of the dataset."""
        print(f"\n\n{'*'*120}\n")
        print(f"Overview of the from {self.transaction_type} Data:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nFirst 5 Rows:")
        print(self.data.head())

    def summary_statistics(self):
        """Display summary statistics for numerical and categorical features."""
        print(f"\n\n{'*'*120}\n")
        if self.transaction_type == 'Bank transaction':
            print(f"No categorical colmns in 'creditcard.csv' and thus summary statistics for categorical data is not applicable")
            return

        print(f"Summary Statistics for numerical features from {self.transaction_type} data:")
        print(self.data.describe())
        print("\n\n")

        print(f"Summary Statistics for categorical features from {self.transaction_type} data:")
        print(self.data.describe(include=[object, 'category']))
        print("\n")

    def retrieve_numerical_columns(self):
        """Return a list of numerical columns."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def identify_missing_values(self):
        """Identify missing values in the dataset."""
        print(f"\n\n{'*'*120}\n")
        print(f"Missing Values in {self.transaction_type} data:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        print("\n")

    def outlier_detection(self):
        """Use box plots to identify outliers in numerical features."""
        print(f"\n\n{'*'*120}\n")
        print(f"Outlier Detection in {self.transaction_type} data:")
        if self.transaction_type == 'e-Commerce transaction':
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
        elif self.transaction_type == 'Bank transaction':
            col_credit_card = 'Amount'
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.data[col_credit_card])
            plt.title(f"Boxplot of {col_credit_card}")
            plt.xlabel(col_credit_card)
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/outliers/outlier_{self.transaction_type}_{col_credit_card}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

    def handle_missing_values(self, strategy="mean", threshold=0.3):
        """
        Handle missing values in the dataset. Features with missing values above the threshold are dropped.
        Otherwise, missing values are handled using imputation (mean, median, or most_frequent).
        """
        print(f"\n\n{'*'*120}\n")
        print(
            f"Handling missing values in {self.transaction_type} data using {strategy} strategy with a threshold of {threshold * 100}%...")

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
        print(f"\n\n{'*'*120}\n")
        initial_row_count = len(self.data)
        print(f"Removing duplicates from {self.transaction_type} data...")
        print(f"Shape of data before removing duplicates: {self.data.shape}")
        self.data.drop_duplicates(inplace=True)
        final_row_count = len(self.data)
        print(f"Duplicates removed. {initial_row_count - final_row_count} rows removed due to duplicacy.")
        print(f"Shape of data after removing duplicates: {self.data.shape}")

    def correct_data_types(self):
        """Convert columns to appropriate data types."""
        print(f"\n\n{'*'*120}\n")
        if self.transaction_type == 'e-Commerce transaction':
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
        elif self.transaction_type == 'Bank transaction' or self.transaction_type == 'Geolocation':
            print(f"Data type correction is not applicable in {self.transaction_type} data.")
        else:
            print("Unknown data")

    def univariate_analysis(self):
        """Perform univariate analysis on numerical columns."""
        print(f"\n\n{'*'*120}\n")
        print(f"Univariate Analysis starting: numerical columns - {self.transaction_type} data...")

        if self.transaction_type == 'e-Comerce transaction':
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
        elif self.transaction_type == 'Geolocation':
            print("Performing Univariate Analysis for Geolocation data...")
            numerical_columns = ['lower_bound_ip_address', 'upper_bound_ip_address']
            for col in numerical_columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[col], kde=True)
                plt.title(f"Univariate Analysis - {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.savefig(
                    f'{BASE_DIR}/notebooks/plots/univariante/numerical/hist_{self.transaction_type}_{col}.png',
                    dpi=300,
                    bbox_inches='tight')
                plt.show()

            print(f"Univariate Analysis for categorical columns in {self.transaction_type} starting...")
            categorical_columns = ['country']  # Assuming country is a categorical column
            for col in categorical_columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=self.data[col])
                plt.title(f"Univariate Analysis - {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.savefig(
                    f'{BASE_DIR}/notebooks/plots/univariante/categorical/countplot_{col}.png',
                    dpi=300,
                    bbox_inches='tight')
                plt.show()

        elif self.transaction_type == 'Bank transaction':
            columns = ['Time', 'Amount']
            for col in columns:
              plt.figure(figsize=(10, 6))
              sns.histplot(self.data[col], kde=True)
              plt.title(f"Univariate Analysis - {col}")
              plt.xlabel(col)
              plt.ylabel("Frequency")
              plt.savefig(
                  f'{BASE_DIR}/notebooks/plots/univariante/numerical/hist_{self.transaction_type}_{col}.png',
                  dpi=300,
                  bbox_inches='tight')
              plt.show()


        """Perform univariante analysis on categorical columns."""
        print(f"Univariate Analysis for categorical columns in {self.transaction_type} starting...")
        if self.transaction_type == "e-Commerce transaction":
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

        elif self.transaction_type == "Bank transaction":
            print(f"There is no categorical colmns in {self.transaction_type} data. So it is not applicable")
        else:
            print("Unknown data")

    def bivariate_analysis(self):
        """Perform bivariate analysis (correlation, pair plots and box plot)."""
        print(f"\n\n{'*'*120}\n")
        # Correlation Heatmap
        print(f"Correlation Heatmap for {self.transaction_type} data:")

        if self.transaction_type == 'e-Commerce transaction':
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
        elif self.transaction_type == 'Bank transaction':
            columns = ['Time', 'Amount', 'Class']

            # Check if all required columns exist in the DataFrame
            if not all(col in self.data.columns for col in columns):
                raise ValueError("One or more required columns are missing from the dataset.")

            # Compute correlation matrix
            corr_matrix = self.data[columns].corr()

            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")

            # Save the figure
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/heatmap_{self.transaction_type}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()


        # Pair Plot (only a subset of columns for better visualization)
        print(f"\nBivariate Analysis - Pair Plot for {self.transaction_type}: Starting...")
        if self.transaction_type == 'e-Commerce transaction':
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
            print(f"Bivariate Analysis - Boxplot for {self.transaction_type} data starting...")
            categorical_columns = ['browser', 'source', 'sex']
            for col in categorical_columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.data[col], y=self.data['class'])
                plt.title(f"Bivariate Analysis - {col} vs class")
                plt.savefig(
                    f'{BASE_DIR}/notebooks/plots/bivariante/categ/boxplot_{col}.png',
                    dpi=300,
                    bbox_inches='tight')
                plt.show()
        elif self.transaction_type == 'Bank transaction':
            print(f"In {self.transaction_type} data, pairpolot and bivariante analysis using boxplot as a categorical data is not that much..already curated data.")

        elif self.transaction_type == 'Geolocation':
            print(f"Performing Bivariate Analysis for Geolocation data...")

            # Correlation between the IP address ranges (if applicable)
            ip_columns = ['lower_bound_ip_address', 'upper_bound_ip_address']
            corr_matrix = self.data[ip_columns].corr()

            # Plot correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap for Geolocation Data")
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/heatmap_{self.transaction_type}.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

            # Boxplot for country vs IP range size (if appropriate)
            self.data['range_size'] = self.data['upper_bound_ip_address'] - self.data['lower_bound_ip_address']
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data['country'], y=self.data['range_size'])
            plt.title(f"Bivariate Analysis - Country vs IP Range Size")
            plt.xlabel('Country')
            plt.ylabel('IP Range Size')
            plt.xticks(rotation=90)
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/bivariante/categ/boxplot_country_ip_range.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()
        else:
            print("Unknown data")

    # Bivariante analysis between categorical variables
    def bivariate_categorical_analysis(self):
        """Perform bivariate analysis between two categorical columns."""
        print(f"\n\n{'*'*120}\n")
        print(f"Bivariate Analysis: contigency table - browser vs source for {self.transaction_type} data: starting")

        if self.transaction_type == 'e-Commerce transaction':
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
        elif self.transaction_type == 'Bank transaction' or self.transaction_type == 'Geolocation':
            print(f"Bivariante analysis for {self.transaction_type} data is not applicabel-No categorical data")
        else:
            print("Unknown data")


    def geolocation_summary_plots(self):
        """Generate summary plots for Geolocation data."""
        print(f"\n\n{'*'*120}\n")
        if self.transaction_type != 'Geolocation':
            print("This method is only applicable for Geolocation data, 'IPAddress_to_Country.csv'. Skipping...")
            return
        print(f"Generating Geolocation Summary Plots for {self.transaction_type} data...")

        if self.transaction_type == 'Geolocation':
            # 1. Bar Plot: Number of IP ranges per country
            print("Generating Bar Plot - Number of IP Ranges per Country...")
            ip_count_per_country = self.data['country'].value_counts()
            plt.figure(figsize=(12, 8))
            sns.barplot(x=ip_count_per_country.index, y=ip_count_per_country.values)
            plt.title("Number of IP Ranges per Country")
            plt.xlabel('Country')
            plt.ylabel('Number of IP Ranges')
            plt.xticks(rotation=45)
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/geolocation/barplot_ip_ranges_per_country.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

            # 2. Choropleth Map: Geographical distribution of IP ranges by country
            print("Generating Choropleth Map - Geographical Distribution of IP Ranges by Country...")
            country_ip_data = self.data.groupby('country').size().reset_index(name='ip_range_count')

            # Load world map data
            url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
            world = gpd.read_file(url)

            # Debugging: Check column names
            print("GeoDataFrame columns:", world.columns)

            # Use the correct column for country names (typically 'ADMIN')
            world = world.rename(columns={'ADMIN': 'country'})

            # Merge data
            merged = world.set_index('country').join(country_ip_data.set_index('country'))

            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            merged.plot(column='ip_range_count', ax=ax, legend=True,
                        legend_kwds={'label': "Number of IP Ranges by Country",
                                    'orientation': "horizontal"})
            plt.title("Geographical Distribution of IP Ranges by Country")
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/geolocation/choropleth_map_ip_ranges.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

            # 3. Scatter Plot: Overlap and gaps between IP ranges
            print("Generating Scatter Plot - Overlap and Gaps Between IP Ranges...")
            self.data['range_size'] = self.data['upper_bound_ip_address'] - self.data['lower_bound_ip_address']
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x=self.data['lower_bound_ip_address'], y=self.data['upper_bound_ip_address'],
                            hue=self.data['country'], palette="Set1", size=self.data['range_size'], sizes=(10, 100))
            plt.title("Scatter Plot - Overlap and Gaps Between IP Ranges")
            plt.xlabel('Lower Bound IP Address')
            plt.ylabel('Upper Bound IP Address')
            plt.savefig(
                f'{BASE_DIR}/notebooks/plots/geolocation/scatterplot_ip_ranges_overlap_gaps.png',
                dpi=300,
                bbox_inches='tight')
            plt.show()

            # **Drop 'range_size' after visualization**
            self.data.drop(columns=['range_size'], inplace=True)

        else:
            print("This method is only applicable for Geolocation data.")



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
        print(f"\n\n{'*'*120}\n")
        if self.transaction_type != 'e-Commerce transaction':
            print("Merging is not applicable for this data. Skipping operation.")
            return


        print(f"Merging data: ip_address from {self.transaction_type} data with ip address range from IpAddress_to_Country.csv")
        # Convert IP addresses to integer format
        self.data['ip_int'] = self.data['ip_address'].apply(self.ip_to_integer)

        # Drop invalid IPs
        self.data = self.data.dropna(subset=['ip_int'])
        self.data['ip_int'] = self.data['ip_int'].astype(
            int)  # Ensure int type

        print(f"Number of valid IPs: {self.data.shape[0]}")

        # Load IP-to-country mapping
        ip_to_country = pd.read_csv(
            f'{BASE_DIR}/IpAddress_to_Country.csv')

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

        print(f"\n\n{'*'*120}\n")
        # Feature engineering is on e-Commerce transaction data the creditcard.csv is already curated.
        if self.transaction_type != 'e-Commerce transaction':
            print("Feature engineering is already applied for 'creditcard.csv' or notnecessary for 'IpAddress_to_Country' data. Skipping operation.")
            return

        print(f"Feature engineering on {self.transaction_type} starting...")

        self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
        self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek

        # Ensure 'purchase_time' is in datetime format
        self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'])

        # Create transaction date column temporarily
        self.data['transaction_date'] = self.data['purchase_time'].dt.date

        # Compute daily transaction amount
        daily_transaction_amount = self.data.groupby(['user_id', 'transaction_date'])['purchase_value'].sum().reset_index()

        # Merge back with original dataset (avoiding column conflict)
        self.data = self.data.merge(daily_transaction_amount, on=['user_id', 'transaction_date'], how='left', suffixes=('', '_24h'))

        # Rename the newly created column properly
        self.data.rename(columns={'purchase_value_24h': 'daily_transaction_amount'}, inplace=True)

        # Drop 'transaction_date' (not needed anymore)
        self.data.drop(columns=['transaction_date'], inplace=True)




        self.data['transaction_velocity'] = (
            self.data['purchase_time'] -
            self.data['signup_time']).dt.total_seconds()
        # self.data['transaction_velocity'] = self.data['transaction_velocity'].apply(
        #     lambda x: max(x, 0))

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
        print(f"\n\n{'*'*120}\n")

        if self.transaction_type == 'Geolocation':
            print("Normalization and scaling is not applicable for Geolocation data.")
            return

        print(f"Normalizing and scaling numerical features on {self.transaction_type}... starting")

        # Initialize scalers
        if self.transaction_type == 'e-Commerce transaction':
            scaler = MinMaxScaler()  # Use MinMaxScaler for e-commerce data
            # List of scalable columns for e-Commerce data
            scalable_columns = [
                'daily_transaction_amount',
                'transaction_velocity',
                'device_shared_count'
            ]
            print(f"Scalable columns for {self.transaction_type} data using {scaler}: {scalable_columns}")
            self.data[scalable_columns] = scaler.fit_transform(self.data[scalable_columns])

        elif self.transaction_type == 'Bank transaction':
            scaler = StandardScaler()  # Use StandardScaler for bank transaction data
            # List of all numerical columns (Time, V1 to V28, Amount)
            scalable_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]  # V1 to V28
            print(f"Scalable columns for {self.transaction_type} data using {scaler}: {scalable_columns}")
            self.data[scalable_columns] = scaler.fit_transform(self.data[scalable_columns])

        else:
            print("Unknown data type")
            return

        print("Normalization and scaling done.")


    def encode_categorical_features(self):
        """Encode categorical features:
        - 'sex' using Label Encoding.
        - 'browser', 'source', 'country', 'age_group' using One-Hot Encoding.
        """
        print(f"\n\n{'*'*120}\n")

        if self.transaction_type != 'e-Commerce transaction':
            print("No categorical columns to be encoded.")
            return

        print(f"Encoding selected categorical features from {self.transaction_type} starting...")

        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown='ignore')

        # Define categorical feature groups
        label_encoding_column = 'sex'
        onehot_encoding_columns = [
            'browser',
            'source',
            'country',
            'age_group']

        # Ensure only existing columns are selected
        onehot_encoding_columns = [
            col for col in onehot_encoding_columns if col in self.data.columns]

        print(f"Label Encoding column: {label_encoding_column}")
        print(f"One-Hot Encoding columns: {onehot_encoding_columns}")

        # Convert to string to handle missing values and prevent issues
        print
        self.data[label_encoding_column] = self.data[label_encoding_column].astype(
            str)
        self.data[onehot_encoding_columns] = self.data[onehot_encoding_columns].astype(
            str)

        print(f"Shape of data after converting into strings: {self.data.shape}")

        # Apply Label Encoding to 'sex'
        if label_encoding_column in self.data.columns:
            self.data[label_encoding_column] = label_encoder.fit_transform(
                self.data[label_encoding_column])
        print(f"Shape of data after label encoding: {self.data.shape}")

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
        print(f"Shape of data after one-hot encoding: {encoded_df.shape}")

        # Drop original categorical columns and merge one-hot encoded data
        self.data = self.data.drop(columns=onehot_encoding_columns)
        self.data = pd.concat([self.data, encoded_df], axis=1)
        print(f"Shape of data after droping original categ and merging columns: {self.data.shape}")

        # Drop any unwanted "_nan" columns and remove NaNs
        self.data = self.data.drop(
            columns=[
                col for col in self.data.columns if "_nan" in col],
            errors="ignore")
        print(f"Shape of data after droping unwanted columns: {self.data.shape}")

        self.data = self.data.dropna()
        print(f"Shape of data after dropping NaNs -that is final: {self.data.shape}")


        print("Categorical encoding completed.")



    def save_processed_data(
            self,
            output_path=None):

        """Save processed data to a CSV file."""

        # Ensure output_path is dynamically set after object creation
        if output_path is None:
            output_path = f"{BASE_DIR}/data/processed_data_{self.transaction_type}.csv"

        print(f"\n\n{'*'*120}\n")
        print(f"Shape of data before saving: {self.data.shape}")
        print(f"Saving processed data to {output_path}...")
        if self.transaction_type == 'Bank transaction':
            print("All creditcard.csv is already curated. No dropping of columns.")
        elif self.transaction_type == 'Geolocation':
            print("Droping of features is not necessary.")
        elif self.transaction_type == 'e-Commerce transaction':

            # Ensure unnecessary columns are droppped
            excluded_columns = ['user_id', 'age', 'device_id', 'signup_time', 'purchase_time', 'purchase_value', 'ip_address', 'ip_int',
                                'lower_bound_ip_address', 'upper_bound_ip_address',
                                'lower_bound_ip_int', 'upper_bound_ip_int']

            self.data = self.data.drop(columns=excluded_columns, errors='ignore')

        else:
            print("Unknown data")

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
        self.geolocation_summary_plots()

        self.merge_datasets_for_geolocation()
        self.feature_engineering()
        self.normalize_and_scale()
        self.encode_categorical_features()
        self.save_processed_data()
