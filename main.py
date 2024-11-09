from models.neuralnetworks import LSTM
from database.utils import create_daily_dataframe
import matplotlib.pyplot as plt

dataset = create_daily_dataframe(
    columns=["Fecha_hoy", "ing_hab"], table="iar_ocupaciones"
)

lstm = LSTM(
    name="prueba",
    created_at="2024-11-03",
    version="1.0",
    metrics=[],
    length=2,
)

lstm.training(dataset=dataset, train=0.65, validation=0.15)

lstm.test(dataset=dataset, test=0.2)
predicted = lstm.predict(dataset=dataset, test=0.2)

predicted_x = dataset.index[-len(predicted) :]
# Create a figure and axis
plt.figure(figsize=(14, 7))

# Plot the original test data
plt.plot(
    dataset.index, dataset["ing_hab"].to_numpy(), label="Original Data", color="blue"
)

# Plot the predicted data
plt.plot(predicted_x, predicted, label="Predicted Data", color="red")

# Adding titles and labels
plt.title("Original vs Predicted Data", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Value", fontsize=14)

# Adding a legend
plt.legend(fontsize=12)

# Optional: Improve date formatting on x-axis
plt.gcf().autofmt_xdate()

# Display the plot
plt.show()
