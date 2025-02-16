"""Flask backend to serve fraud detection data through API endpoints."""

from flask import Flask, jsonify
import pandas as pd


app = Flask(__name__)

# Load fraud data
df = pd.read_csv("Fraud_Data.csv")

# Load the processed data to analyze frauds by geographical location
df_processed = pd.read_csv("processed_data.csv")

print(df_processed.columns)


# fraud_counts = df_processed[df_processed["class"] == 1].iloc[:, 5:].sum().to_dict()

# Ensure that the 'purchase_time' column is in datetime format
df['purchase_time'] = pd.to_datetime(df['purchase_time'])

# Extractin features
df["hour_of_day"] = df["purchase_time"].dt.hour
df["day_of_week"] = df["purchase_time"].dt.dayofweek

@app.route("/", methods=["GET"])
def home():
  return jsonify({"message": "Fraud API running"}), 200


@app.route("/summary", methods=["GET"])

def summary():
  total_transaction = len(df)
  total_fraud_cases = df[df["class"] == 1].shape[0]
  fraud_percentage = (total_fraud_cases / total_transaction) * 100
  total_features = list(df.columns)

  summary_data = {
      "total_transactions": total_transaction,
      "total_fraud_cases": total_fraud_cases,
      "fraud_percentage": round(fraud_percentage, 2),
      "total_features": total_features,
  }

  return jsonify(summary_data)


@app.route("/fraud_by_country", methods=["GET"])
def fraud_by_country():
    # fraud_counts = df_processed[df_processed["class"] == 1]["country"].value_counts().to_dict()
    fraud_counts = df_processed[df_processed["class"] == 1].filter(like="country_").sum().to_dict()
    return jsonify(fraud_counts)


@app.route("/fraud_trend", methods=["GET"])
def fraud_trend():
    fraud_cases_per_day = df.groupby('purchase_time')["class"].sum().reset_index()
    return jsonify(fraud_cases_per_day.to_dict(orient="records"))

@app.route("/fraud_trend_by_day", methods=["GET"])
def fraud_trend_by_day():
    fraud_cases_per_day = df.groupby("hour_of_day")["class"].sum().reset_index()
    return jsonify(fraud_cases_per_day.to_dict(orient="records"))

@app.route("/fraud_trend_by_week", methods=["GET"])
def fraud_trend_by_week():
    fraud_cases_per_week = df.groupby("day_of_week")["class"].sum().reset_index()
    return jsonify(fraud_cases_per_week.to_dict(orient="records"))

@app.route("/fraud_by_device", methods=["GET"])
def fraud_by_device():
    fraud_counts = df[df["class"] == 1]["device_id"].value_counts().to_dict()
    return jsonify(fraud_counts)

@app.route("/fraud_by_browser", methods=["GET"])
def fraud_by_browser():
    fraud_counts = df[df["class"] == 1]["browser"].value_counts().to_dict()
    return jsonify(fraud_counts)

if __name__ == "__main__":
    app.run(debug=True)

