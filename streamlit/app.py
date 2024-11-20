import streamlit as st

# Set the page configuration (optional)
st.set_page_config(
    page_title="Time Series Analysis App",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add a title and subtitle
st.title("Time Series Analysis App")
st.subheader("Explore various time series forecasting models")

# Add an image at the top
# st.image("images/overview.png", use_column_width=True)

# Write some introductory text
st.write(
    """
This application allows you to explore different time series forecasting models, including:
"""
)

# Add images alongside explanations
st.header("Available Models")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("LSTM Model")
    # st.image("images/lstm_model.png", use_column_width=True)
    st.write(
        """
    The Long Short-Term Memory (LSTM) network is a type of recurrent neural network capable of learning order dependence in sequence prediction problems.
    """
    )

with col2:
    st.subheader("Holt-Winters Model")
    # st.image("images/holt_winters_model.png", use_column_width=True)
    st.write(
        """
    The Holt-Winters method is a time series forecasting technique that includes level, trend, and seasonal components.
    """
    )
with col3:
    st.subheader("Prophet Model")
    # st.image("images/holt_winters_model.png", use_column_width=True)
    st.write(
        """
    Prophet is an open-source time series forecasting tool developed by Facebook's Core Data Science team.
    """
    )

st.write(
    """
👉 **Use the sidebar to navigate to different models and learn more about them.**
"""
)

st.header("App Information")

st.write(
    """
Each model includes:

- **Graphs Related to Each Model**: Visual representations of the model's performance and data patterns.
- **Predictions**: The model's forecasted values based on historical data.

**How the App Works:**

- **Fetching Models from Azure Blob Storage**: The pre-trained models are stored in Azure Blob Storage and are fetched dynamically when needed.
- **Making Predictions with Example Dataset**: The app uses an example dataset to demonstrate how the models make predictions.
- **Making Prediction for Future dates: The pre-trained models contain a method that predicts the future days (LSTM, Holt-Winters) or specific dates (Prophet) value per day.**

This setup allows for scalable storage and efficient retrieval of models, ensuring you always have access to the latest versions without needing to store large files locally.
"""
)
