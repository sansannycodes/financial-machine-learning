"""T5 - European call option pricing with Black-Scholes and ML models."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def black_scholes_call(S, K, T, r, sigma):
    """Price a European call option using the Black-Scholes formula."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Build a reproducible synthetic option-pricing dataset.
rng = np.random.default_rng(42)
n = 2000
S = rng.uniform(70, 150, n)
K = rng.uniform(70, 150, n)
T = rng.uniform(0.1, 2.0, n)
r = rng.uniform(0.01, 0.08, n)
sigma = rng.uniform(0.10, 0.60, n)

prices = np.array([black_scholes_call(s, k, t, rate, vol)
                   for s, k, t, rate, vol in zip(S, K, T, r, sigma)])

data = pd.DataFrame({"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "call_price": prices})
X = data[["S", "K", "T", "r", "sigma"]]
y = data["call_price"]

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    "SVR": make_pipeline(StandardScaler(), SVR(C=100, epsilon=0.01, gamma="scale")),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    mae = -scores.mean()
    rows.append({"Model": name, "CV_MAE": mae})

results = pd.DataFrame(rows).sort_values("CV_MAE")
print(results.to_string(index=False))

# Fit one model and show common regression metrics on the full generated sample.
final_model = models["Gradient Boosting"]
final_model.fit(X, y)
pred = final_model.predict(X)
print("\nGradient Boosting training metrics")
print(f"MAE : {mean_absolute_error(y, pred):.6f}")
print(f"MSE : {mean_squared_error(y, pred):.6f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, pred)):.6f}")
print(f"R2  : {r2_score(y, pred):.6f}")

results.to_csv("model_comparison.csv", index=False)
