import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from dotenv import load_dotenv
from datetime import timedelta

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

predicted = model.predict(dataset=DATASET, test=0.2)
predicted_x = DATASET[-len(predicted) :]

st.title("Prophet Test Results")

fig, ax = plt.subplots(figsize=(14, 7))

# Plot the original test data
ax.plot(
    DATASET["Day"], DATASET["ing_hab"].to_numpy(), label="Original Data", color="blue"
)
# Plot the predicted data
ax.plot(predicted_x["Day"], predicted["yhat"], label="Predicted Data", color="red")

# Adding titles and labels using the correct methods
ax.set_title("Original vs Predicted Data", fontsize=16)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Value", fontsize=14)

# Adding a legend
ax.legend(fontsize=12)

# Optional: Improve date formatting on x-axis
fig.autofmt_xdate()

# Display the plot in Streamlit
st.pyplot(fig)

st.write(f"Metrics: ", model.test(dataset=DATASET, test=0.2))

last_date = DATASET["Day"].iloc[-1]

start_date = st.date_input(
    "Start date to predict:", value=last_date + timedelta(days=1)
)
end_date = st.date_input("End date to predict:", value=last_date + timedelta(days=30))

if st.button("Predict"):
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        st.success(f"Prediction period: {start_date} to {end_date}")
        df = model.predict_dates(start_date, end_date)
        st.write(df)

        df = pd.concat(
            [DATASET.tail(1).rename(columns={"Day": "ds", "ing_hab": "yhat"}), df],
            ignore_index=True,
        )
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot the original test data
        ax.plot(
            DATASET["Day"],
            DATASET["ing_hab"].to_numpy(),
            label="Original Data",
            color="blue",
        )
        # Plot the predicted data
        ax.plot(df["ds"], df["yhat"].to_numpy(), label="Predicted Data", color="red")

        # Adding titles and labels using the correct methods
        ax.set_title("Predicted Data", fontsize=16)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)

        # Adding a legend
        ax.legend(fontsize=12)

        # Optional: Improve date formatting on x-axis
        fig.autofmt_xdate()

        # Display the plot in Streamlit
        st.pyplot(fig)
