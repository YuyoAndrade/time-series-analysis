import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_sequences(data, seq_length):
    x, y = [], []
    column = data.columns[-1]
    for i in range(len(data) - seq_length):
        x.append(data[column][i : i + seq_length])
        y.append(data[column][i + seq_length])

    return np.array(x), np.array(y)


def normalize(data):
    scaler = StandardScaler()
    fitted = scaler.fit(data[["ing_hab"]])
    scaled = fitted.transform(data[["ing_hab"]])
    df_scaled = pd.DataFrame(scaled, columns=["ing_hab_scaled"])
    df = pd.concat([data, df_scaled], axis=1)
    return df, fitted


def get_train_validation_data(data, train, validation=0):
    num_train_samples = int(train * len(data))
    num_validation_samples = int(validation * len(data))

    print("num_train_samples:", num_train_samples)
    print("num_validation_samples:", num_validation_samples)

    return (num_train_samples, num_validation_samples)


def get_test_data(data, test):
    num_test_samples = int(test * len(data))
    print("num_test_samples:", num_test_samples)
    return num_test_samples
