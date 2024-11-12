from .neuralnetworks import LSTM
from .holt_winters import HOLT_WINTERS
from datetime import date

TRAIN = 0.65
VALIDATION = 0.15
TEST = 0.2

MODELS_TO_CREATE = {
    "LSTM": LSTM(
        name="LSTM",
        created_at=date.today(),
        version="1.0",
        length=2,
    ),
    "HOLT-WINTERS": HOLT_WINTERS(
        name="HOLT-WINTERS",
        created_at=date.today(),
        version="1.0",
        seasonal_period=365,
    ),
}
