# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dropout, Dense


class NeuralNetwork:
    def __init__(self, model_type, name, created_at, version, metrics=None):
        self.model_type = model_type
        self.name = name
        self.created_at = created_at
        self.version = version
        self.metrics = metrics

    def metrics(self):
        pass

    def predict(self):
        pass

    def LSTM(self, length):
        model = Sequential()
        model.add(InputLayer(input_shape=(length, 1)))
        # First LSTM layer
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))

        # Dense layer
        model.add(Dense(units=25, activation="relu"))
        model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(units=1))  # Linear activation is default for regression

        # metrics = ""
        # self.metrics = self.
        # return self
        return self
