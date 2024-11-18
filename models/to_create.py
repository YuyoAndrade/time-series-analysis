from .neuralnetworks import LSTM
from .holt_winters import HOLT_WINTERS
from datetime import date
from dotenv import load_dotenv
import os

load_dotenv(override=True)

TRAIN = 0.65
VALIDATION = 0.15
TEST = 0.2

LSTM_MODEL = os.getenv("LSTM_MODEL")
HOLT_WINTERS_MODEL = os.getenv("HOLT_WINTERS_MODEL")

MODELS_TO_CREATE = {
    "LSTM": LSTM(
        name=LSTM_MODEL,
        created_at=date.today(),
        version="1.0",
        length=2,
    ),
    "HOLT-WINTERS": HOLT_WINTERS(
        name=HOLT_WINTERS_MODEL,
        created_at=date.today(),
        version="1.0",
        seasonal_period=365,
    ),
}
