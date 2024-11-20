import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os


from dotenv import load_dotenv
from datetime import timedelta

from database.connection import create_blob_client
from database.utils import download_blob_to_file, azure_daily_get_dataframe


def graph_plotly(hist, actual_test, predicted_test, future):
    # Create the base figure
    figplot = go.Figure()

    # Add static traces (historical and actual test data)
    figplot.add_trace(
        go.Scatter(
            x=hist["Day"],
            y=hist["ing_hab"],
            mode="lines",
            name="Información histórica",
            line=dict(width=2, color="white"),
        )
    )
    figplot.add_trace(
        go.Scatter(
            x=actual_test["Day"],
            y=actual_test["ing_hab"],
            mode="lines",
            name="Valores reales",
            line=dict(width=2, color="gray"),
            visible=True,
        )
    )

    # Add animated traces (initialized with empty data)
    figplot.add_trace(
        go.Scatter(
            x=predicted_test["Day"][:1],
            y=predicted_test["ing_hab"][:1],
            mode="lines",
            name="Predicción prueba",
            line=dict(width=2, color="orange"),
        )
    )
    figplot.add_trace(
        go.Scatter(
            x=future["Day"][:1],
            y=future["ing_hab"][:1],
            mode="lines",
            name="Predicción Futura",
            line=dict(width=2, color="green"),
        )
    )

    # Calculate total frames
    len_orange = len(predicted_test)
    len_green = len(future)
    total_frames = len_orange + len_green

    # Generate frames
    frames = []
    for i in range(1, total_frames + 1):
        # Update the orange line data
        if i <= len_orange:
            orange_x = predicted_test["Day"][:i]
            orange_y = predicted_test["ing_hab"][:i]
        else:
            orange_x = predicted_test["Day"]
            orange_y = predicted_test["ing_hab"]

        # Update the green line data
        green_index = i - len_orange
        if green_index > 0:
            green_x = future["Day"][:green_index]
            green_y = future["ing_hab"][:green_index]
        else:
            green_x = []
            green_y = []

        # Append the frame
        frames.append(
            go.Frame(
                data=[
                    # Only update the animated traces (indices 2 and 3)
                    go.Scatter(
                        x=orange_x,
                        y=orange_y,
                        mode="lines",
                        line=dict(width=2, color="orange"),
                        name="Predicción prueba",
                    ),
                    go.Scatter(
                        x=green_x,
                        y=green_y,
                        mode="lines",
                        line=dict(width=2, color="green"),
                        name="Predicción Futura",
                    ),
                ],
                traces=[2, 3],  # Indicate which traces to update
            )
        )

    # Assign frames to the figure
    figplot.frames = frames

    # Update layout with 'Simulate' and 'Pause' buttons
    figplot.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Simulate",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=10, redraw=True), fromcurrent=False
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
                        ],
                    ),
                ],
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
        title="Predicción de Ingreso Habitacional de Enero 1 2024 a Diciembre 31 2024",
        xaxis_title="Fecha",
        yaxis_title="Ingreso USD",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Display the figure
    st.plotly_chart(figplot)


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
    download_path="./models/tmp/",
    to_path="/streamlit/tmp/",
)

st.subheader("Prophet Model Implementation")

with open(f"/streamlit/tmp/{PROPHET_MODEL}.pkl", "rb") as f:
    model = pickle.load(f)

predicted = model.predict(dataset=DATASET, test=0.2)
predicted_x = DATASET[-len(predicted) :]

dfHistorico = DATASET[DATASET["Day"] < predicted_x["Day"].iloc[0]]
dfActualTest = DATASET[DATASET["Day"] >= predicted_x["Day"].iloc[0]]
dfPrediccionTest = pd.DataFrame(
    {"Day": predicted_x["Day"], "ing_hab": predicted["yhat"]}
)

st.subheader("Prophet Test Results")

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

        df = pd.concat(
            [DATASET.tail(1).rename(columns={"Day": "ds", "ing_hab": "yhat"}), df],
            ignore_index=True,
        )

        dfPrediccionFuturo = df.rename(columns={"ds": "Day", "yhat": "ing_hab"})

        graph_plotly(dfHistorico, dfActualTest, dfPrediccionTest, dfPrediccionFuturo)

        st.write(df)
