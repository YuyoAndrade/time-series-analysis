from .neuralnetworks import LSTM
from datetime import date

TRAIN = 0.65
VALIDATION = 0.15
TEST = 0.2

MODELS_TO_CREATE = {
    "LSTM": LSTM(
        name="LSTM",
        created_at=date.today(),
        version="1.0",
        metrics=[],
        length=2,
    )
}
