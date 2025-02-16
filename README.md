# Credit Card and e-Commerce Fraud Detection

> This fraud detection project aims to build improved detection systems using Machine Learning (ML) and Deep Learning (DL) techniques to identify fraudulent e-commerce and credit card transactions. The project involves data cleaning, preprocessing, feature engineering, and encoding. Exploratory Data Analysis (EDA) is conducted to gain insights into individual features and the relationships between variables. Traditional ML models and modern DL architectures (CNN and LSTM) are implemented and evaluated. Additionally, `mlflow` is used to track the models' runtime. The project is primarily implemented using Python and is containerized with Docker, with visualizations provided by the `dash` Python package.
---

## Built With

- **Major Language:** Python 3
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, torch
- **Tools & Technologies:** Jupyter Notebook, Google Colab, Git, GitHub, Gitflow, VS Code


## Demonstration and Website

[Deployment link]()

---


## Getting Started

You can clone this project, use it freely, and contribute to its development.

1. Clone the repository:
   ```bash
   git clone https://github.com/amare1990/fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fraud-detection
   ```
3. Create a Python virtual environment:
   ```bash
   python3 -m venv venv-name
   ```
   Replace `venv-name` with your preferred environment name.

4. Activate the virtual environment:
   - **Linux/macOS:**
     ```bash
     source venv-name/bin/activate
     ```
   - **Windows:**
     ```bash
     venv-name\Scripts\activate
     ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Run the project:
   - To automate the workflow and execute the pipeline end-to-end:
     ```bash
     python src/src.py
     ```
   - To experiment with individual workflow components, open **Jupyter Notebook** and run each notebook file. The filenames are carefully selected to match their corresponding scripts.

### Prerequisites

Ensure you have the following installed:
- Python (version 3.8.10 or higher)
- pip
- Git
- VS Code

### Dataset

To build and evaluate the fraud detection system, the following datasets are utilized:

- **`Fraud_data.csv`** – Contains transaction records labeled as fraudulent or legitimate. I used this data to build models.
- **`IP_Address_To_Country.csv`** – Provides geolocation data, mapping IP addresses to specific countries.
- **`Creditcard.csv`** – Includes credit card transaction details with fraud labels.

These datasets support transaction analysis, feature engineering, and model training, ensuring a comprehensive approach to fraud detection.

## Folder Structure

```bash
fraud_detection/
├── .gitignore                # Git ignore file to exclude unnecessary files
├── README.md                 # Documentation for your project
├── .githubworkflows/
│   └── pylint.yml            # GitHub Actions workflow for running pylint checks
├── fraud_detection_api/      # API serving the fraud detection models
│   ├── scripts/               # Contains scripts for models too
    ├── serve_model_api.py    # API to serve models for predictions
│   └── Docker
    ├── requirements.txt       # Copied packages from the project
│   └── test_api.py            # To test the Flask api


        # Additional files for the API
├── fraud_detection_dashboard/  # Folder for the dashboard app
│   ├── app.py                # Main dashboard application script
│   └── dashboard.py        # Additional files for the dashboard
    ├── static           # Main dashboard application script
│   └── templates
    ├── Fraud_Dta.csv              # Main dashboard application script
│   └── processed_data.csv
    └── requirements.txt

├── data/                     # Folder containing the dataset
│   └── Fraud_Data.csv         # Raw data
    └── processed_data.csv     # Your dataset, processed and ready for use
├── models/                   # Folder where the models are stored
│   ├── Logistic Regression.pkl  # Serialized Logistic Regression model
│   ├── Decision Tree.pkl       # Serialized Decision Tree model
│   ├── Random Forest.pkl       # Serialized Random Forest model
│   ├── Gradient Boosting.pkl   # Serialized Gradient Boosting model
│   ├── MLP.pkl                # Serialized MLP Classifier model
│   ├── CNN.pth                # Serialized CNN model
│   ├── LSTM.pth               # Serialized LSTM model
│   └── RNN.pth                # Serialized RNN model
├── scripts/                   # Folder containing utility scripts
    ├── __init__.py
│   ├── data_analysis_preprocessing.py
    ├── model_training.py
│   └── pipeline_model_building_processes.py
    ├── model_explainability.py
│   └── model_explainer.py  # Contains the pipeline function for the model explainability module
    ├── cnn.py  # Contains CNNModel class
│   └── lstm.py  # Contains the LSTMModel class
    ├── rnn.py  # Contains the RNNModel class
│   └── dl_wrapper.py  # Contains wrapper class for the DL models
├── src/                       # Source code for your project
│   ├── src.py          # Code for training, testing, etc.
    ├── __init__.py
├── notebooks/                 # Folder for Jupyter Notebooks│
│   ├── plots/                 # Folder where EDA plots, SHAP and LIME plots are saved
│   └── data_processor.ipynb   # To run data_analysis_preprocessing.py
    └── model_building.ipynb   # To run model_trainin.py processes
    └── model_explainer.ipynb  # To run model_explainability.py processes
├── mlruns/                    # Folder for MLflow experiment logs
└── v-fraud-detec/             # Virtual environment for fraud detection

```
---

## Project Requirements
- Git, GitHub setup, adding `pylint' in the GitHub workflows
- Data Analysis and Preprocessing
- Model Building and Training
- Model Explainability
- Model Deployment and API Development
- Build a Dashboard with Flask and Dash


### GitHub Action and Following Python Coding Styles
- The `pylint` linters are added in the `.github/workflows` direcory of the repo, which is triggered while creating GitHub pull requests.
- Make it to check when Pull request is created
- Run `pylint scripts/*.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive scripts/*.py` to automatically fix some linters errors


### Data Analysis and Preprocessing

The **data analysis and preprocessing** phase is crucial for ensuring the quality and reliability of the fraud detection model. The following steps were performed:

#### **1. Data Cleaning**
- Handled missing values by applying appropriate imputation techniques.
- Removed duplicate transactions to prevent bias in model training.
- Standardized and normalized numerical features to ensure consistency.
- Transformed data: Converted `transaction_time` and `signup_time` categorical data into date time format.
- Detected outliers using boxplot visualizations

#### **2. Exploratory Data Analysis (EDA)**
- Analyzed feature distributions and transaction patterns.
- Identified correlations between features using heatmaps and scatter plots.
- Examined fraudulent vs. legitimate transaction characteristics.
- Visualized time-series trends and anomalies in transaction data.

#### **3. Feature Engineering**
- Extracted new meaningful features from timestamps, geolocation, and transaction history.
- Applied one-hot encoding, label encoding, and frequency encoding to categorical variables.
- Engineered aggregated features such as **average transaction amount per user** and **transaction frequency over time**.
- Reduced dimensionality using **Principal Component Analysis (PCA)** and **feature selection techniques**.

#### **4. Data Splitting & Preprocessing for Model Training**

- Split the dataset into **training (80%)** and **testing (20%)** sets.
- Scaled numerical features using **Min-Max Scaling** and **Standardization**.
- Converted categorical features into numerical representations.


### Model Building

This project implements a fraud detection model using both traditional machine learning and deep learning techniques. The model-building process consists of data preparation, training, evaluation, and model saving.
The core functionality of model building, training, evaluation, saving and tracking experiments are implemented in the `scripts/model_training.py` script. To pipeline all processes of model building, training, evaluation, savining and tracking the model experiments with `mlflow`,`pipeline_model_building_processes.py` script is implemented and run to view results. Below is a detailed breakdown of each step:

#### 1. Data Preparation

- Before training, the dataset undergoes preprocessing to enhance model performance:

- Feature Selection: Irrelevant columns such as device_id, signup_time, purchase_time, and IP-related fields are excluded.

- Data Splitting: The dataset is divided into training (80%) and testing (20%) sets using train_test_split from scikit-learn.

#### 2. Training Traditional Machine Learning Models

- Several machine learning models are trained using scikit-learn:

   - Logistic Regression

   - Decision Tree Classifier

   - Random Forest Classifier

   - Gradient Boosting Classifier

   - MLP Classifier (Neural Network with one hidden layer)

- Each model is trained and evaluated based on the following metrics:

   - Accuracy

   - Precision

   - Recall

   - F1-score

   - Confusion Matrix

   - Classification Report

#### 3. Training Deep Learning Models

- The deep learning models are implemented using PyTorch, leveraging their strengths for sequential and pattern-based fraud detection. The models include:

   - CNN (Convolutional Neural Network) – Captures spatial relationships in data.

   - RNN (Recurrent Neural Network) – Effective for sequence-based fraud patterns.

   - LSTM (Long Short-Term Memory Network) – Handles long-term dependencies in sequential data.

- Training Configuration:

   - Loss Function: Binary Cross-Entropy Loss (BCELoss)

   - Optimizer: Adam optimizer with a default learning rate of 0.001

   - Batch Size: 32 (configurable)

   - Epochs: 10 (adjustable as needed)

- After training, performance metrics similar to those used in traditional models are computed. Additionally, training loss curves are plotted and saved to assess model convergence.

#### 4. Model Saving and Storage

   - Trained models are saved for future inference and evaluation:

   - Traditional Models: Stored using pickle.

   - Deep Learning Models: Saved using torch.save() in .pth format.

#### 5. Experiment Tracking with MLflow

- To track model performance and parameter tuning, MLflow is integrated into the workflow:

- Setting Up MLflow:

   - pip install mlflow

- Tracking Experiments:

   - Log hyperparameters, metrics, and model versions.

   - Store training history and visualization plots.

- Running the MLflow UI:

   - mlflow ui

   - Allows visualization of model performance over multiple experiments.

- This structured approach ensures systematic training, evaluation, and tracking for both traditional and deep learning models in fraud detection.


### Model Explainability


This module provides explainability tools for machine learning models using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations). It supports a variety of models, including traditional machine learning models (like Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting) and deep learning models (such as MLP, CNN, LSTM, and RNN). This allows users to interpret model predictions and understand feature contributions.

## Features

- **SHAP Explainer**: SHAP values provide a unified measure of feature importance and contribution to model predictions.
  - Summary plots for feature importance.
  - Force plots for individual instance prediction explanations.
  - Dependence plots to show how feature values affect predictions.

- **LIME Explainer**: LIME provides local explanations for individual predictions, helping users interpret black-box models.
  - Generates visual plots for feature importance for each instance.

## Installation

Before using this module, make sure you have the following Python libraries installed:

```bash
pip install shap lime
```

## Class: `ModelExplainability`

### Overview
The `ModelExplainability` class provides methods to generate explainability plots using SHAP and LIME for a variety of machine learning models. It handles both traditional ML models and deep learning models, utilizing SHAP and LIME's power to explain model predictions.

### Methods

#### `__init__(self, model, X_train, X_test, feature_names, base_dir)`
Initializes the explainability module with the provided model and dataset.

- **model**: Trained ML model (can be scikit-learn or deep learning model).
- **X_train**: Training data used for explainability.
- **X_test**: Test data for model prediction and explanation.
- **feature_names**: List of feature names.
- **base_dir**: Base directory for saving generated plots.

#### `shap_explain(self, sample_index=0, feature_index=0)`
Generates SHAP explanations for the provided model and dataset.

- **sample_index**: Index of the sample in the test set to generate individual SHAP explanations.
- **feature_index**: Index of the feature for the SHAP dependence plot.

Generates the following SHAP visualizations:
- **Summary plot**: Visualizes feature importance across all samples.
- **Force plot**: Provides an individual explanation for a specific sample's prediction.
- **Dependence plot**: Shows the effect of a specific feature on the model's output.

#### `lime_explain(self, sample_index=0, num_features=10)`
Generates a LIME explanation for a specific test sample.

- **sample_index**: Index of the sample to generate LIME explanation.
- **num_features**: Number of features to include in the LIME explanation.

Generates the following LIME visualizations:
- **Explanation plot**: Shows how the features of the sample contribute to the model's prediction.

## Example Usage

```python
import pickle
import pandas as pd
import torch
from model_explainability import ModelExplainability

# Load dataset
df = pd.read_csv("path/to/your/data.csv")
X = df.drop(columns=["target"])
X_train, X_test = X.iloc[:int(0.8 * len(X))], X.iloc[int(0.8 * len(X)):]
feature_names = X_train.columns.tolist()

# Load trained model
with open("path/to/your/model.pkl", 'rb') as f:
    model = pickle.load(f)

# Initialize explainability
explainer = ModelExplainability(model, X_train, X_test, feature_names, "path/to/save/plots")

# SHAP Analysis
explainer.shap_explain(sample_index=5)

# LIME Analysis
explainer.lime_explain(sample_index=5)
```

---

## Pipeline: `pipeline_model_explainability`

### Overview
The `pipeline_model_explainability` function automates the process of loading models, datasets, and generating SHAP/LIME explanations for multiple models.

### Steps:
1. **Load Models**: Loads different models from specified file paths, including Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, MLP, CNN, LSTM, and RNN.
2. **Load Dataset**: Reads the dataset and prepares it by dropping unnecessary columns and splitting it into training and testing sets.
3. **Explainability**: For each model, the pipeline initializes the `ModelExplainability` class and runs both SHAP and LIME explainability methods.

### Example Usage

```python
from model_explainability_pipeline import pipeline_model_explainability

# Run the model explainability pipeline
pipeline_model_explainability()
```

### Required Files:
1. **Models**: Ensure that your models are saved in `.pkl` format (for traditional ML models) or `.pth` format (for deep learning models like CNN, LSTM, RNN).
2. **Dataset**: The dataset should be in `.csv` format, with the target column labeled as `"class"` and the unnecessary columns dropped as required.



## Model Deployment and API Development



```markdown
## fraud_etection_aPI

This project provides a Flask API for fraud detection using machine learning (ML) and deep learning (DL) models, such as Logistic Regression, Decision Trees, CNN, LSTM, and RNN. It's Dockerized for easy deployment.

### Requirements
- Python>3.8
- Docker

### Directory setup
- Navigate to the root directort
  - cd fraud_detection
- Navigate the api endpoint project
  -cd fraud_detection_api


Install dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Docker Setup
1. **Build the Docker Image**:
   ```bash
   docker build -t fraud-detection-api .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 5000:5000 fraud-detection-api
   ```

3. You may set up directories while running a docker, by mounting both data and models (Change the directory of you have.)
   - > sudo docker run -p 5000:5000 \
-v "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data:/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/data" \
-v "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/models:/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/models" \
fraud_detection_api
```



### API Endpoints

- **GET /**: Returns a message indicating the API is running.

- **POST /predict**: Predicts fraud on a transaction. Example request:
  ```json
  {
    "purchase_value": 250.5,
    "age": 32,
    "browser": "Chrome",
    "ip_address": "192.168.1.1",
    ...
  }
  ```
  Response:
  ```json
  {
    "model_used": "Logistic Regression",
    "prediction": 0,
    "fraud_probability": 0.12
  }
  ```

### Models
The API uses the following pre-trained models:
- **Machine Learning**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP Classifier
- **Deep Learning**: CNN, LSTM, RNN

### Testing
Use the provided `test_api.py` script to test the API locally.

---



## Build a Dashboard with Flask and Dash

```markdown

This section explains how to build a dashboard using Flask for the backend and Dash for the frontend to visualize fraud detection trends and statistics.

### Backend with Flask (`app.py`)

The Flask application serves data through various API endpoints that provide information about fraud cases. The endpoints include fraud statistics by different categories such as geographic location, time, device, and browser.

#### Key Endpoints:
- **GET /**: Returns a message indicating that the API is running.
- **GET /summary**: Provides a summary of fraud detection data, including the total number of transactions, fraud cases, fraud percentage, and features.
- **GET /fraud_by_country**: Returns the number of fraud cases grouped by country.
- **GET /fraud_trend**: Returns the fraud cases trend over time.
- **GET /fraud_trend_by_day**: Provides the trend of fraud cases by hour of the day.
- **GET /fraud_trend_by_week**: Shows the trend of fraud cases by day of the week.
- **GET /fraud_by_device**: Returns the number of fraud cases per device.
- **GET /fraud_by_browser**: Provides the fraud cases per browser.

#### Running Flask Backend:
To run the Flask app, use the following command:
```bash
python app.py
```

This will start the Flask server at `http://127.0.0.1:5000`, where you can access the endpoints.

### Frontend with Dash (`dashboard.py`)

The Dash application visualizes the fraud data fetched from the Flask API. It displays various interactive charts such as geographical distribution of fraud cases, fraud trends over time, and fraud counts by device and browser.

#### Features of the Dashboard:
- **Summary Box**: Displays the total number of transactions, fraud cases, fraud percentage, and features in the dataset.
- **Geographic Distribution**: Shows fraud cases by country using a choropleth map.
- **Fraud Trend (Time)**: Displays a line chart of fraud cases over time.
- **Fraud Trend (Day of Week)**: Visualizes fraud cases by day of the week.
- **Fraud Trend (Hour of Day)**: Shows fraud cases by hour of the day.
- **Fraud by Device**: A bar chart displaying the number of fraud cases per device.
- **Fraud by Browser**: A bar chart displaying the number of fraud cases per browser.

#### Running the Dash App:
To run the Dash app, use the following command:
```bash
python dashboard.py
```

This will start the Dash server at `http://127.0.0.1:8050`, where the fraud detection dashboard can be accessed.

### Data Flow
1. **Flask Backend**: The Flask app serves the fraud data through API endpoints.
2. **Dash Frontend**: The Dash app makes API requests to the Flask backend to retrieve the data and visualize it using Plotly charts.

#### Fetching Data in Dash:
The data is fetched from the Flask API using HTTP requests. For example, the fraud data by country is retrieved as follows:
```python
fraud_by_country = requests.get("http://127.0.0.1:5000/fraud_by_country").json()
```

### Dependencies:
Make sure you have the required Python packages installed. You can install them using `pip`:
```bash
pip install flask dash requests pandas plotly
``


```

---

## Future Works

The next steps involve building an end-to-end fraud detection pipeline:
- **Balancing classes using SMOTified+GAN**


> #### You can gain more insights by running the jupter notebook and view plots.
---

### More information
- You can refer to [this link](https://drive.google.com/file/d/1aZKOSMJHP8vytMt3DrpnHAZAGm5A15Xp/view) to gain more insights about the reports of this project results.

## Authors

👤 **Amare Kassa**

- GitHub: [@githubhandle](https://github.com/amare1990)
- Twitter: [@twitterhandle](https://twitter.com/@amaremek)
- LinkedIn: [@linkedInHandle](https://www.linkedin.com/in/amaremek/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/amare1990/fraud-detection.git/issues).

## Show your support

Give a ⭐️ if you like this project, and you are welcome to contribute to this project!

## Acknowledgments

- Hat tip to anyone whose code was referenced to.
- Thanks to the 10 academy and Kifiya financial instituion that gives me an opportunity to do this project

## 📝 License

This project is [MIT](./LICENSE) licensed.
