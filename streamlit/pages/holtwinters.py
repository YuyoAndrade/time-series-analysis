import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from dotenv import load_dotenv

from database.connection import create_blob_client
from database.utils import download_blob_to_file, azure_daily_get_dataframe

st.title("Holt-Winters Model")

st.write(
    """
The Holt-Winters method is a time series forecasting technique that includes
level, trend, and seasonal components.

**Key Features:**

- Captures seasonality.
- Suitable for short-term forecasting.
- Easy to implement and interpret.

"""
)

load_dotenv(override=True)

BLOB_MODEL_CONTAINER = os.getenv("BLOB_MODEL_CONTAINER")
HOLT_WINTERS_MODEL = os.getenv("HOLT_WINTERS_MODEL")

DATASET = azure_daily_get_dataframe("daily")

blob_client = create_blob_client()

download_blob_to_file(
    blob_service_client=blob_client,
    model_name=HOLT_WINTERS_MODEL,
    container_name=BLOB_MODEL_CONTAINER,
    download_path="/models/tmp/",
)


def predict_graph(dataset, model, next):
    predicted = model.predict(next=next)

    predicted_x = dataset.index[-len(predicted) :]

    st.subheader(f"Holt-Winters Test Results of next {next} days.")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the original test data
    ax.plot(
        dataset.index,
        dataset["ing_hab"].to_numpy(),
        label="Original Data",
        color="blue",
    )

    # Plot the predicted data
    ax.plot(predicted_x, predicted, label="Predicted Data", color="red")

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
    return


st.subheader("Holt-Winters Model Implementation")

with open(f"/streamlit/models/tmp/{HOLT_WINTERS_MODEL}.pkl", "rb") as f:
    model = pickle.load(f)

predict_graph(dataset=DATASET, model=model, next=round(len(DATASET) * 0.2))

next_days = st.text_input("Prediction X next days:")

# Button to execute an action
if st.button("Predict"):
    predict_graph(dataset=DATASET, model=model, next=int(next_days))
