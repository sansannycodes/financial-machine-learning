"""T7 - Treasury yield-curve prediction examples.

CSV format expected:
Date,1M,3M,6M,1Y,2Y,5Y,10Y,30Y

The script demonstrates moving-average forecasting, ARIMA forecasting for
individual maturities, Nelson-Siegel curve fitting, and simple Vasicek/CIR
interest-rate simulations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from statsmodels.tsa.arima.model import ARIMA

DATA_FILE = "treasury_yields.csv"
MATURITIES = [1/12, 3/12, 6/12, 1, 2, 5, 10, 30]
COLUMNS = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]


df = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
df = df.set_index("Date")
yields = df[COLUMNS].dropna()

# 1. Moving-average forecast
window = 5
ma_forecast = yields.rolling(window).mean().iloc[-1]
print("Moving-average forecast:")
print(ma_forecast)

# 2. ARIMA forecast for each maturity
arima_forecast = {}
for col in COLUMNS:
    series = yields[col]
    model = ARIMA(series, order=(1, 1, 1)).fit()
    arima_forecast[col] = float(model.forecast(steps=1).iloc[0])

print("\nARIMA one-step forecast:")
print(pd.Series(arima_forecast))

# 3. Nelson-Siegel model
def nelson_siegel(tau, beta0, beta1, beta2, lam):
    x = tau / lam
    loading = (1 - np.exp(-x)) / x
    curvature = loading - np.exp(-x)
    return beta0 + beta1 * loading + beta2 * curvature


def fit_nelson_siegel(tau, observed):
    def residuals(params):
        return nelson_siegel(tau, *params) - observed

    initial = [observed[-1], observed[0] - observed[-1], 0.0, 1.0]
    result = least_squares(residuals, initial, bounds=([-10, -10, -10, 0.01], [20, 20, 20, 20]))
    return result.x


last_curve = yields.iloc[-1].values
params = fit_nelson_siegel(np.array(MATURITIES), last_curve)
fitted_curve = nelson_siegel(np.array(MATURITIES), *params)

print("\nNelson-Siegel parameters [level, slope, curvature, lambda]:")
print(params)
print("Fitted curve:")
print(pd.Series(fitted_curve, index=COLUMNS))

# 4. Simple Vasicek and CIR simulations
def vasicek(r0, kappa, theta, sigma, steps=252, dt=1/252, seed=42):
    rng = np.random.default_rng(seed)
    rates = np.empty(steps + 1)
    rates[0] = r0
    for t in range(steps):
        z = rng.normal()
        rates[t + 1] = rates[t] + kappa * (theta - rates[t]) * dt + sigma * np.sqrt(dt) * z
    return rates


def cir(r0, kappa, theta, sigma, steps=252, dt=1/252, seed=42):
    rng = np.random.default_rng(seed)
    rates = np.empty(steps + 1)
    rates[0] = r0
    for t in range(steps):
        z = rng.normal()
        diffusion = sigma * np.sqrt(max(rates[t], 0)) * np.sqrt(dt) * z
        drift = kappa * (theta - rates[t]) * dt
        rates[t + 1] = max(rates[t] + drift + diffusion, 0)
    return rates


r0 = float(yields["10Y"].iloc[-1]) / 100
vasicek_path = vasicek(r0, kappa=0.5, theta=r0, sigma=0.02)
cir_path = cir(r0, kappa=0.5, theta=r0, sigma=0.08)

print(f"\nVasicek final simulated rate: {vasicek_path[-1] * 100:.4f}%")
print(f"CIR final simulated rate:     {cir_path[-1] * 100:.4f}%")

pd.DataFrame({
    "maturity": COLUMNS,
    "moving_average": ma_forecast.values,
    "arima": [arima_forecast[c] for c in COLUMNS],
    "nelson_siegel": fitted_curve,
}).to_csv("yield_curve_predictions.csv", index=False)
