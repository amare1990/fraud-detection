"""Flask backend to serve fraud detection data through API endpoints."""

from flask import Flask, jsonify
import pandas as pd


app = Flask(__name__)

# Load fraud data

df = pd.read_csv("Fraud_Data.csv")


@app.route("/", methods=["GET"])
def home():
  return jsonify({"message": "Fraud API running"}), 200


@app.route("/summary", methods=["GET"])

def summary():
  total_transaction = len(df)
  total_fraud_cases = df[df["class"] == 1].shape[0]
  fraud_percentage = (total_fraud_cases / total_transaction) * 100

  summary_data = {
      "total_transactions": total_transaction,
      "total_fraud_cases": total_fraud_cases,
      "fraud_percentage": round(fraud_percentage, 2),
  }

  return jsonify(summary_data)

@app.route("/fraud_trend", methods=["GET"])
def fraud_trend():
    fraud_cases_per_day = df.groupby("date")["class"].sum().reset_index()
    return jsonify(fraud_cases_per_day.to_dict(orient="records"))

@app.route("/fraud_by_device", methods=["GET"])
def fraud_by_device():
    fraud_counts = df[df["class"] == 1]["device"].value_counts().to_dict()
    return jsonify(fraud_counts)

@app.route("/fraud_by_browser", methods=["GET"])
def fraud_by_browser():
    fraud_counts = df[df["class"] == 1]["browser"].value_counts().to_dict()
    return jsonify(fraud_counts)

if __name__ == "__main__":
    app.run(debug=True)

