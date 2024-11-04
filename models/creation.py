from neuralnetworks import NeuralNetwork
from database.utils import get_specifics
import pandas as pd

lstm = NeuralNetwork(
    model_type="LSTM",
    name="prueba",
    created_at="2024-11-03",
    version="1.0",
    metrics=[],
)

result = get_specifics(["Fecha_hoy", "ing_hab"], "iar_ocupaciones")

df = pd.DataFrame(result.fetchall(), columns=result.keys())
print(df.head())
print(lstm.LSTM(length=10))
