from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dropout, Dense, LayerNormalization
from tensorflow.keras.layers import LSTM as layerLSTM

from .model import Model
from .utils import get_train_validation_data, get_test_data, create_sequences


class LSTM(Model):
    def __init__(
        self,
        name,
        created_at,
        version,
        length=2,
        metrics=None,
        model_type=None,
        model=None,
    ):
        super().__init__(name, created_at, version, metrics, model_type, model)
        self.model_type = "LSTM"
        self.length = length

    def build_model(self):
        model = Sequential()
        model.add(InputLayer(shape=(self.length, 1)))
        # First LSTM layer
        model.add(layerLSTM(units=8))
        # Output Layer
        model.add(Dense(units=1, activation="linear"))

        self.model = model
        return model

    def training(self, dataset, train, validation=0):
        model = self.model
        if not model:
            model = self.build_model()

        x, y = create_sequences(dataset, self.length)
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=0.00000001),
            metrics=["mean_absolute_error"],
        )
        model.summary()

        num_train, num_validation = get_train_validation_data(
            data=dataset, train=train, validation=validation
        )

        x_train = x[:][:num_train]
        x_validation, y_validation = (
            x[:][num_train : num_train + num_validation],
            y[:][num_train : num_train + num_validation],
        )
        y_train = y[:][:num_train]

        x_train = x_train.reshape((x_train.shape[0], self.length, 1))
        x_validation = x_validation.reshape((x_validation.shape[0], self.length, 1))

        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

        model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=100,
            batch_size=4,
            callbacks=callbacks,
            shuffle=False,
        )
        self.model = model
        return True

    def test(self, dataset, test):
        num_test = get_test_data(data=dataset.to_numpy(), test=test)
        x, y = create_sequences(dataset, self.length)

        x_test = x[:][-num_test:]
        y_test = y[:][-num_test:]

        x_test = x_test.reshape((x_test.shape[0], self.length, 1))

        return self.model.evaluate(x_test, y_test)  # Evaluate Model

    def predict(self, dataset, test):
        num_test = get_test_data(data=dataset.to_numpy(), test=test)
        x, _ = create_sequences(dataset, self.length)

        x_test = x[:][-num_test:]

        x_test = x_test.reshape((x_test.shape[0], self.length, 1))
        return self.model.predict(x_test)  # Return Prediction Values
