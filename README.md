# Credit Card and e-Commerce Fraud Detection

> This fraud detection project aims to build improved detection systems using Machine Learning (ML) and Deep Learning (DL) techniques to identify fraudulent e-commerce and credit card transactions. The project involves data cleaning, preprocessing, feature engineering, and encoding. Exploratory Data Analysis (EDA) is conducted to gain insights into individual features and the relationships between variables. Traditional ML models and modern DL architectures (CNN and LSTM) are implemented and evaluated. Additionally, MLOps is used to track the models' runtime. The project is primarily implemented using Python.

## Built With

- **Major Language:** Python 3
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, torch
- **Tools & Technologies:** Jupyter Notebook, Google Colab, Git, GitHub, Gitflow, VS Code

## Demonstration and Website

[Deployment link]()

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

- **`Fraud_data.csv`** – Contains transaction records labeled as fraudulent or legitimate.
- **`IP_Address_To_Country.csv`** – Provides geolocation data, mapping IP addresses to specific countries.
- **`Creditcard.csv`** – Includes credit card transaction details with fraud labels.

These datasets support transaction analysis, feature engineering, and model training, ensuring a comprehensive approach to fraud detection.

### Project Requirements
- Git, GitHub setup, adding `pylint' in the GitHub workflows
- Data Analysis and Preprocessing
- Model Building and Training
- Model Explainability
- Model Deployment and API Development
- Build a Dashboard with Flask and Dash


#### GitHub Action and Following Python Coding Styles
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

This project implements a fraud detection model using various machine learning and deep learning techniques. The model building process involves data preparation, training, evaluation, and saving the models. Below are the steps taken to build the models:

#### Data Preparation

- The dataset undergoes preprocessing before training:

- Excludes irrelevant columns such as device_id, signup_time, purchase_time, and IP-related fields.

- Splits data into training and testing sets using an 80-20 ratio with train_test_split.

#### Training Traditional Machine Learning Models

- Several machine learning models are trained using scikit-learn:

- > Logistic Regression

      Decision Tree Classifier

      Random Forest Classifier

      Gradient Boosting Classifier

      MLP Classifier (Neural Network with one hidden layer)

- For each model, the following evaluation metrics are computed:

- > Accuracy

      Precision

      Recall

      F1-score

      Confusion Matrix

      Classification Report

#### Training Deep Learning Models

- The deep learning models are implemented using PyTorch and include:

- > CNN (Convolutional Neural Network)

   RNN (Recurrent Neural Network)

   LSTM (Long Short-Term Memory Network)

- Each model is trained using:

- > Binary Cross-Entropy Loss (BCELoss) as the loss function.

   Adam Optimizer with a default learning rate of 0.001.

   Mini-batch training with a batch size of 32.

   10 epochs (configurable).

- Performance metrics similar to traditional models are computed after training. Additionally, training loss curves are plotted and saved.

#### Model Saving

- The trained models are stored for future use:

- Traditional models are saved using pickle.

- Deep learning models (CNN, RNN, LSTM) are saved using torch.save() in .pth format.

- This structured approach ensures efficient training and evaluation of both machine learning and deep learning models for fraud detection.

#### Tracking and Experimenting With `mflow`

- Setting Up MLflow: `pip install mlflow`.
- Tracking Experiments – Logging parameters, metrics, and models.
- Running the MLflow UI – Viewing experiment results.
- Integration with Your Code – Adding MLflow to your existing training pipeline



### Feature Works

The next steps involve building an end-to-end fraud detection pipeline:

- **Model Explainability** – Use SHAP, LIME, and feature importance analysis to interpret model decisions.
- **Model Deployment and API Development** – Deploy the trained model using Flask and FastAPI for real-time fraud detection.
- **Build a Dashboard with Flask and Dash** – Develop an interactive web-based dashboard for visualizing fraud detection insights.


> #### You can gain more insights by running the jupter notebook and view plots.


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
