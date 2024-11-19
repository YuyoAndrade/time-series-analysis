import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from dotenv import load_dotenv
from datetime import date

from database.connection import create_blob_client
from database.utils import download_blob_to_file, azure_daily_get_dataframe

st.title("Prophet Model")

st.write(
    """
Prophet is an open-source time series forecasting tool developed by Facebook's Core Data Science team. It is designed to make forecasting easier and more accurate for non-experts in time series data.

**Key Features:**

- Handles seasonality and trend components automatically.
- Supports modeling of holidays and special events.
- Robust to missing data and outliers.
- Provides intuitive and interpretable parameters.
"""
)

load_dotenv(override=True)

BLOB_MODEL_CONTAINER = os.getenv("BLOB_MODEL_CONTAINER")
PROPHET_MODEL = os.getenv("PROPHET_MODEL")

DATASET = azure_daily_get_dataframe("daily")

blob_client = create_blob_client()

download_blob_to_file(
    blob_service_client=blob_client,
    model_name=PROPHET_MODEL,
    container_name=BLOB_MODEL_CONTAINER,
    download_path="/models/tmp/",
)

st.subheader("Prophet Model Implementation")

with open(f"/streamlit/models/tmp/{PROPHET_MODEL}.pkl", "rb") as f:
    model = pickle.load(f)

start_date = st.date_input("Start date to predict:", value=date.today())
end_date = st.date_input("End date to predict:", value=date.today())

if st.button("Predict"):
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        st.success(f"Prediction period: {start_date} to {end_date}")
        df = model.predict(start_date, end_date)
        st.write(df)
