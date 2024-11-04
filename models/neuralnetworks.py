from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dropout, Dense
from tensorflow.keras.layers import LSTM as layerLSTM

from .model import Model
from .utils import get_train_validation_data, get_test_data, create_sequences


class LSTM(Model):
    def __init__(
        self,
        name,
        created_at,
        version,
        length=10,
        metrics=None,
        model_type=None,
    ):
        self.model_type = "LSTM"
        self.length = length

    def build_model(self):
        model = Sequential()
        model.add(InputLayer(shape=(self.length, 1)))
        # First LSTM layer
        model.add(layerLSTM(units=150, return_sequences=True))
        model.add(Dropout(0.2))

        # Second LSTM layer
        model.add(layerLSTM(units=150))
        model.add(Dropout(0.2))

        # Dense layer
        model.add(Dense(units=75, activation="relu"))
        model.add(Dropout(0.2))

        # Dense layer
        model.add(Dense(units=75, activation="relu"))
        model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(units=1))

        return model

    def training(self, dataset):
        dataset = dataset.to_numpy()
        model = self.build_model()
        x, y = create_sequences(dataset, self.length)
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=0.01),
            metrics=["mean_absolute_error"],
        )
        model.summary()

        num_train, num_validation = get_train_validation_data(data=dataset, train=0.8)

        x_train = x[:][:num_train]
        x_validation, y_validation = (
            x[:][num_train : num_train + num_validation],
            y[:][num_train : num_train + num_validation],
        )
        y_train = y[:][:num_train]

        x_train = x_train.reshape((x_train.shape[0], 10, 1))
        x_validation = x_validation.reshape((x_validation.shape[0], 10, 1))

        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

        model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            shuffle=False,
        )
        return model

    def test(self, model, dataset, test):
        num_test = get_test_data(data=dataset.to_numpy(), test=test)
        x, y = create_sequences(dataset.to_numpy(), self.length)
        x_test = x[:][-num_test:]
        y_test = y[:][-num_test:]

        return model.evaluate(x_test, y_test)
