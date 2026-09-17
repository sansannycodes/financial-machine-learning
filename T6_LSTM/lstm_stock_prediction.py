"""T6 - LSTM time-series prediction.

Downloads historical data with yfinance, scales the close price, creates
60-step sequences and trains a small LSTM model.
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

TICKER = "AAPL"
START = "2018-01-01"
END = "2025-01-01"
WINDOW = 60


data = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
close = data["Close"].dropna().values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close)

split = int(len(scaled) * 0.8)
train_data = scaled[:split]
test_data = scaled[split - WINDOW:]


def make_sequences(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)


X_train, y_train = make_sequences(train_data, WINDOW)
X_test, y_test = make_sequences(test_data, WINDOW)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

pred_scaled = model.predict(X_test, verbose=0)
pred = scaler.inverse_transform(pred_scaled).ravel()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual")
plt.plot(pred, label="Predicted")
plt.title(f"{TICKER} Stock Price - LSTM")
plt.xlabel("Test time step")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
