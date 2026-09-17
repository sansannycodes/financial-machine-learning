"""T4 - Stock price prediction using ARIMA.

Expected CSV columns: Date, Stock_1, Stock_2, ...
The tutorial uses Stock_1 as the prediction target.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = "stock_data.csv"
TARGET = "Stock_1"


df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")
series = df[TARGET].dropna()

# Stationarity check
adf_stat, p_value, *_ = adfuller(series)
print(f"ADF statistic: {adf_stat:.4f}")
print(f"ADF p-value: {p_value:.4f}")

# First differencing is used before fitting the ARIMA model.
train_size = int(len(series) * 0.8)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

model = ARIMA(train, order=(7, 1, 7))
result = model.fit()
forecast = result.forecast(steps=len(test))

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("\nARIMA(7,1,7) results")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

comparison = pd.DataFrame({"Actual": test, "Predicted": forecast}, index=test.index)
comparison.to_csv("arima_predictions.csv")
print("Saved predictions to arima_predictions.csv")
