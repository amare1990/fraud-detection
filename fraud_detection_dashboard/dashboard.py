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

# Fetch fraud by country data
fraud_by_country = requests.get("http://127.0.0.1:5000/fraud_by_country").json()
fraud_by_country_df = pd.DataFrame(list(fraud_by_country.items()), columns=["country", "fraud_cases"])


fraud_trend = pd.DataFrame(requests.get("http://127.0.0.1:5000/fraud_trend").json())
fraud_trend_by_day = pd.DataFrame(requests.get("http://127.0.0.1:5000/fraud_trend_by_day").json())
fraud_trend_by_week = pd.DataFrame(requests.get("http://127.0.0.1:5000/fraud_trend_by_week").json())
fraud_by_device = requests.get("http://127.0.0.1:5000/fraud_by_device").json()
fraud_by_browser = requests.get("http://127.0.0.1:5000/fraud_by_browser").json()


# dashboard layout
app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),

    # Summary Boxes
    html.Div(children=[
        html.Div(f"Total Transactions: {summary_data['total_transactions']}", className="box"),
        html.Div(f"Total Fraud Cases: {summary_data['total_fraud_cases']}", className="box"),
        html.Div(f"Fraud Percentage: {summary_data['fraud_percentage']}%", className="box"),
        html.Div(f"Total features: {summary_data['total_features']}", className="box"),
    ], className="summary-boxes"),


    dcc.Graph(
        figure=px.choropleth(
            fraud_by_country_df,
            locations="country",  # Country names
            locationmode="country names",  # Ensure compatibility with country names
            color="fraud_cases",  # Color intensity based on fraud cases
            title="Geographic Distribution of Fraud Cases",
            color_continuous_scale="Reds",
            labels={"fraud_cases": "Fraud Cases"},
        )
    ),


     # Fraud Trend Line Chart General
    dcc.Graph(
        figure=px.line(
            fraud_trend,
            x="purchase_time",
            y="class",
            title="Fraud Cases Over Time",
            labels={"purchase_time": "Date", "class": "Number of Fraud Cases"},
            markers=True
        )
    ),

    # Fraud Trend Line Chart by Week
    dcc.Graph(
        figure=px.line(
            fraud_trend_by_week,
            x="day_of_week",
            y="class",
            title="Fraud Cases Over Week",
            labels={"Day of week": "Day", "class": "Number of Fraud Cases"},
            markers=True
        )
    ),

    # # Fraud Trend Line Chart by Day
    dcc.Graph(
        figure=px.line(
            fraud_trend_by_day,
            x="hour_of_day",
            y="class",
            title="Fraud Cases Over day",
            labels={"Hour of day": "Hour", "class": "Number of Fraud Cases"},
            markers=True
        )
    ),

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


