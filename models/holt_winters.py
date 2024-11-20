import pandas as pd
import numpy as np
import logging

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

from .model import Model
from .utils import get_train_validation_data, get_test_data, create_sequences


class HOLT_WINTERS(Model):
    def __init__(
        self,
        name,
        created_at,
        version,
        seasonal_period=365,
        metrics=None,
        model_type=None,
        model=None,
    ):
        super().__init__(name, created_at, version, metrics, model_type, model)
        self.model_type = "HOLT-WINTERS"
        self.seasonal_period = seasonal_period

    def build_model(self):
        return self.model

    def train(self, dataset, train, validation=0):
        x, _ = create_sequences(dataset, 1)

        num_train, _ = get_train_validation_data(
            data=dataset, train=train + validation, validation=0
        )

        x_train = x[:][:num_train]

        model = ExponentialSmoothing(
            x_train,
            trend="mul",
            seasonal="mul",
            seasonal_periods=self.seasonal_period,
            use_boxcox=True,
            initialization_method="estimated",
        )

        self.model = model.fit()
        return True

    def test(self, dataset, test):
        test_data = dataset.to_numpy()
        num_test = get_test_data(data=test_data, test=test)
        _, y = create_sequences(dataset, 1)

        y_test = y[-num_test:]

        test_forecast = self.model.forecast(num_test)
        test_series = pd.Series(test_forecast)

        rmse = mean_squared_error(y_test, test_series) ** 0.5
        logging.info("Root Mean Squared Error (RMSE):", rmse)

        mape = np.mean(np.abs((y_test - test_series) / y_test))
        logging.info("Mean Absolute Percent Error:", mape)

        mae = np.mean(np.abs(y_test - test_series))
        logging.info("Mean Absolute Error:", mae)

        self.metrics = {"RMSE": rmse, "MAPE": mape, "MAE": mae}
        return self.metrics

    def predict(self, next):
        return self.model.forecast(next)
