import streamlit as st

import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from database.connection import create_blob_client
from database.utils import download_blob_to_file, azure_daily_get_dataframe

st.title("LSTM Model")

st.write(
    """
The Long Short-Term Memory (LSTM) network is a type of recurrent neural network
capable of learning order dependence in sequence prediction problems.

**Key Features:**

- Handles long-term dependencies.
- Effective for time series forecasting.
- Prevents vanishing gradient problem.

"""
)

st.subheader("LSTM Model Implementation")

DATASET = azure_daily_get_dataframe("daily")

load_dotenv(override=True)

BLOB_MODEL_CONTAINER = os.getenv("BLOB_MODEL_CONTAINER")
LSTM_MODEL = os.getenv("LSTM_MODEL")

blob_client = create_blob_client()

download_blob_to_file(
    blob_service_client=blob_client,
    model_name=LSTM_MODEL,
    container_name=BLOB_MODEL_CONTAINER,
    download_path="/models/tmp/",
)

with open(f"/streamlit/models/tmp/{LSTM_MODEL}.pkl", "rb") as f:
    model = pickle.load(f)

predicted = model.predict(dataset=DATASET, test=0.2)

predicted_x = DATASET.index[-len(predicted) :]


st.title("LSTM Test Results")

fig, ax = plt.subplots(figsize=(14, 7))

# Plot the original test data
ax.plot(
    DATASET.index, DATASET["ing_hab"].to_numpy(), label="Original Data", color="blue"
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
