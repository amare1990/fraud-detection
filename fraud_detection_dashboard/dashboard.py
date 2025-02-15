"""Script to visualize fraud trends using Dash. """


import dash

from dash import dcc, html
import requests
import pandas as pd
import plotly.express as px


# Initialize Dash APP

app = dash.Dash(__name__)


# Fetch data from the local server
summary_data = requests.get("http://127.0.0.1:5000/summary").json()
fraud_trend = pd.DataFrame(requests.get("http://127.0.0.1:5000/fraud_trend").json())
fraud_by_device = requests.get("http://127.0.0.1:5000/fraud_by_device").json()
fraud_by_browser = requests.get("http://127.0.0.1:5000/fraud_by_browser").json()


# dashboard layout
# Layout
app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),

    # Summary Boxes
    html.Div(children=[
        html.Div(f"Total Transactions: {summary_data['total_transactions']}", className="box"),
        html.Div(f"Total Fraud Cases: {summary_data['total_fraud_cases']}", className="box"),
        html.Div(f"Fraud Percentage: {summary_data['fraud_percentage']}%", className="box"),
    ], className="summary-boxes"),

    # Fraud by Device Bar Chart
    dcc.Graph(
        figure=px.bar(
            x=list(fraud_by_device.keys()),
            y=list(fraud_by_device.values()),
            title="Fraud Cases by Device",
            labels={"x": "Device", "y": "Number of Fraud Cases"},
        )
    ),

    # Fraud by Browser Bar Chart
    dcc.Graph(
        figure=px.bar(
            x=list(fraud_by_browser.keys()),
            y=list(fraud_by_browser.values()),
            title="Fraud Cases by Browser",
            labels={"x": "Browser", "y": "Number of Fraud Cases"},
        )
    ),
])

# Run Dash App
if __name__ == "__main__":
    app.run_server(debug=True)


