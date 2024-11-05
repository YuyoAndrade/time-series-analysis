from models.neuralnetworks import LSTM
from database.utils import create_dataframe

dataset = create_dataframe(columns=["Fecha_hoy", "ing_hab"], table="iar_ocupaciones")

lstm = LSTM(
    name="prueba",
    created_at="2024-11-03",
    version="1.0",
    metrics=[],
)

model = lstm.training(dataset=dataset)
print(model)
