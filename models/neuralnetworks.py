from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.layers import LSTM as layerLSTM
from datetime import date, timedelta

import pandas as pd
import numpy as np
import logging

from .model import Model
from .utils import get_train_validation_data, get_test_data, create_sequences, normalize


class LSTM(Model):
    def __init__(
        self,
        name,
        created_at,
        version,
        length=7,
        metrics=None,
        model_type=None,
        model=None,
        scaler=None,
    ):
        super().__init__(name, created_at, version, metrics, model_type, model)
        self.model_type = "LSTM"
        self.length = length
        self.scaler = scaler

    def build_model(self):
        model = Sequential()
        model.add(InputLayer(shape=(self.length, 1)))
        # First LSTM layer
        model.add(layerLSTM(units=32))
        # Output Layer
        model.add(Dense(units=1, activation="linear"))

        self.model = model
        return model

    def train(self, dataset, train, validation=0):
        model = self.model
        if not model:
            model = self.build_model()

        dataset, scaler = normalize(data=dataset)
        dataset = dataset.drop("ing_hab", axis=1)

        self.scaler = scaler

        x, y = create_sequences(dataset, self.length)
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=0.0001),
            metrics=["mean_absolute_percentage_error"],
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
            callbacks=callbacks,
            shuffle=False,
        )
        self.model = model
        return True

    def test(self, dataset, test):
        num_test = get_test_data(data=dataset.to_numpy(), test=test)

        dataset, _ = normalize(data=dataset)
        dataset = dataset.drop("ing_hab", axis=1)

        x, y = create_sequences(dataset, self.length)

        x_test = x[:][-num_test:]
        y_test = y[:][-num_test:]

        x_test = x_test.reshape((x_test.shape[0], self.length, 1))
        test_predictions = self.model.predict(x_test).flatten()
        test_results = pd.DataFrame(
            data={"Test Predictions": test_predictions, "Actuals": y_test}
        )
        original_scale_data = self.scaler.inverse_transform(test_results)

        test_results_original = pd.DataFrame(
            original_scale_data,
            columns=["Test Predictions", "Actuals"],
            index=test_results.index,
        )

        predictions = test_results_original["Test Predictions"]
        actuals = test_results_original["Actuals"]

        mape = np.mean(np.abs((actuals - predictions) / actuals))
        logging.info("Mean Absolute Percent Error:", mape)

        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        logging.info("Root Mean Squared Error (RMSE):", rmse)

        mae = np.mean(np.abs(actuals - predictions))
        logging.info("Mean Absolute Error:", mae)

        self.metrics = {"RMSE": rmse, "MAPE": mape, "MAE": mae}
        return self.metrics

    def predict(self, dataset, test=0):
        num_test = get_test_data(data=dataset.to_numpy(), test=test)

        dataset, _ = normalize(data=dataset)
        dataset = dataset.drop("ing_hab", axis=1)

        x, _ = create_sequences(dataset, self.length)

        x_test = x[:][-num_test:]

        x_test = x_test.reshape((x_test.shape[0], self.length, 1))

        return self.scaler.inverse_transform(
            pd.DataFrame(data={"Predictions": self.model.predict(x_test).flatten()})
        )

    def predict_next(self, dataset, next, last_date=date.today()):
        df = pd.DataFrame(columns=["Day", "ing_hab"])

        last = dataset
        for i in range(next):
            last_transformed = self.scaler.transform(last.reshape(-1, 1))
            last_transformed = last_transformed.reshape(
                (last_transformed.shape[1], last_transformed.shape[0], 1)
            )
            predicted = self.model.predict(last_transformed)
            predicted = self.scaler.inverse_transform(predicted)
            last_date = last_date + timedelta(days=1)
            logging.info(
                f"Predicted on: {last} ----- Result: Day: {last_date}, Value: {predicted}"
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        data=[
                            [
                                last_date,
                                predicted[0][0],
                            ]
                        ],
                        columns=["Day", "ing_hab"],
                    ),
                ],
                ignore_index=True,
            )
            last = np.append(last, predicted[0][0])
            last = last[1:]
        df["Day"] = pd.to_datetime(df["Day"])
        return df
