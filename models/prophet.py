import pandas as pd
import numpy as np
import logging

from prophet import Prophet
from sklearn.metrics import mean_squared_error

from .model import Model
from .utils import get_train_validation_data, get_test_data


class PROPHET(Model):
    def __init__(
        self,
        name,
        created_at,
        version,
        metrics=None,
        model_type=None,
        model=None,
    ):
        super().__init__(name, created_at, version, metrics, model_type, model)
        self.model_type = "PROPHET"

    def build_model(self):
        self.model = Prophet()
        return self.model

    def train(self, dataset, train, validation=0):
        dataset = dataset.rename(
            columns={dataset.columns[0]: "ds", dataset.columns[1]: "y"}
        )

        num_train, _ = get_train_validation_data(
            data=dataset.to_numpy(), train=train + validation, validation=0
        )

        x_train = dataset.head(num_train)

        model = self.model

        self.model = model.fit(x_train)
        return True

    def test(self, dataset, test):
        test_data = dataset.to_numpy()
        num_test = get_test_data(data=test_data, test=test)

        y_test = dataset.tail(num_test)[dataset.columns[1]]

        test_future = self.model.make_future_dataframe(periods=num_test)
        test_forecast = self.model.predict(test_future)

        rmse = mean_squared_error(y_test, test_forecast["yhat"][-num_test:]) ** 0.5
        logging.info("Root Mean Squared Error (RMSE):", rmse)

        mape = np.mean(abs(y_test - test_forecast["yhat"][-num_test:]) / y_test)
        logging.info("Mean Absolute Percent Error:", mape)

        mae = np.mean(np.abs(y_test - test_forecast["yhat"][-num_test:]))
        logging.info("Mean Absolute Error:", mae)

        self.metrics = {"RMSE": rmse, "MAPE": mape, "MAE": mae}
        return self.metrics

    def predict(self, dataset, test):
        test_data = dataset.to_numpy()
        num_test = get_test_data(data=test_data, test=test)
        test_future = self.model.make_future_dataframe(periods=num_test)
        test_forecast = self.model.predict(test_future)

        return test_forecast[["ds", "yhat"]][-num_test:]

    def predict_dates(self, start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date)
        date_df = pd.DataFrame({"ds": date_range})

        return self.model.predict(df=date_df)[["ds", "yhat"]]
