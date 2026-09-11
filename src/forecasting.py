"""
Forecasting Module — Spice Price Dynamics BTP
==============================================
SARIMA, LSTM, and Hybrid (SARIMA + LSTM) forecasting models
for spice price prediction with MAPE evaluation.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Statsmodels SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

# PyTorch LSTM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROCESSED_DIR  = Path(__file__).resolve().parent.parent / "data" / "processed"
PRECOMPUTED_DIR = Path(__file__).resolve().parent.parent / "data" / "precomputed"
MODELS_DIR     = Path(__file__).resolve().parent.parent / "models"


# --- SARIMA ------------------------------------------------------------------

def fit_sarima(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
               train_pct: float = 0.8) -> dict:
    """
    Fit SARIMA model with train/test split and return forecast + metrics.
    """
    n = len(series)
    train_size = int(n * train_pct)
    train, test = series.iloc[:train_size], series.iloc[train_size:]

    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=200)

        # Forecast on test period
        forecast = fitted.forecast(steps=len(test))
        forecast.index = test.index

        # Metrics
        mape = mean_absolute_percentage_error(test, forecast) * 100
        rmse = np.sqrt(mean_squared_error(test, forecast))

        return {
            "model_type": "SARIMA",
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "train_size": train_size,
            "test_size": len(test),
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "aic": round(fitted.aic, 2),
            "bic": round(fitted.bic, 2),
            "train_dates": [str(train.index[0]), str(train.index[-1])],
            "test_dates": [str(test.index[0]), str(test.index[-1])],
            "actual": test.values.tolist(),
            "predicted": forecast.values.tolist(),
            "dates": [str(d) for d in test.index],
            "fitted_model": fitted,
        }
    except Exception as e:
        print(f"    SARIMA error: {e}")
        return None


# --- LSTM --------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Simple LSTM for time series forecasting."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data: np.ndarray, lookback: int = 12) -> tuple:
    """Create input sequences for LSTM."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def fit_lstm(series: pd.Series, lookback: int = 12, epochs: int = 100,
             hidden_size: int = 64, train_pct: float = 0.8) -> dict:
    """
    Fit LSTM model with train/test split and return forecast + metrics.
    """
    values = series.values.reshape(-1, 1).astype(np.float32)

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # Create sequences
    X, y = create_sequences(scaled, lookback)

    n = len(X)
    train_size = int(n * train_pct)

    X_train = torch.FloatTensor(X[:train_size])
    y_train = torch.FloatTensor(y[:train_size])
    X_test = torch.FloatTensor(X[train_size:])
    y_test_scaled = y[train_size:]

    # Model
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Predict
    model.eval()
    with torch.no_grad():
        if len(X_test) > 0:
            predictions_scaled = model(X_test).numpy()
        else:
            return None

    # Inverse transform
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    actual = scaler.inverse_transform(y_test_scaled).flatten()

    # Metrics
    mape = mean_absolute_percentage_error(actual, predictions) * 100
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    # Date alignment: test dates start at lookback + train_size
    test_start_idx = lookback + train_size
    test_dates = series.index[test_start_idx:test_start_idx + len(actual)]

    return {
        "model_type": "LSTM",
        "lookback": lookback,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "train_size": train_size,
        "test_size": len(actual),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "actual": actual.tolist(),
        "predicted": predictions.tolist(),
        "dates": [str(d) for d in test_dates],
        "model_state": model.state_dict(),
        "scaler": scaler,
    }


# --- Hybrid (SARIMA + LSTM) -------------------------------------------------

def fit_hybrid(series: pd.Series, train_pct: float = 0.8) -> dict:
    """
    Hybrid approach:
    1. Fit SARIMA to capture linear trend + seasonality
    2. Fit LSTM on the SARIMA residuals to capture nonlinear patterns
    3. Combine forecasts: hybrid = SARIMA_forecast + LSTM_residual_forecast
    """
    n = len(series)
    train_size = int(n * train_pct)
    train = series.iloc[:train_size]

    # Step 1: SARIMA
    try:
        sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False, enforce_invertibility=False)
        sarima_fitted = sarima_model.fit(disp=False, maxiter=200)

        # In-sample residuals
        sarima_insample = sarima_fitted.fittedvalues
        residuals = train - sarima_insample
        residuals = residuals.dropna()

        # SARIMA out-of-sample forecast
        sarima_forecast = sarima_fitted.forecast(steps=n - train_size)
        sarima_forecast.index = series.index[train_size:]
    except Exception as e:
        print(f"    Hybrid SARIMA step failed: {e}")
        return None

    # Step 2: LSTM on residuals
    if len(residuals) < 25:
        # Not enough residuals for LSTM, return SARIMA-only
        test = series.iloc[train_size:]
        mape = mean_absolute_percentage_error(test, sarima_forecast) * 100
        rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
        return {
            "model_type": "Hybrid (SARIMA only — insufficient residuals for LSTM)",
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "actual": test.values.tolist(),
            "predicted": sarima_forecast.values.tolist(),
            "dates": [str(d) for d in test.index],
            "train_size": train_size,
            "test_size": len(test),
        }

    try:
        lookback = min(6, len(residuals) // 4)
        res_values = residuals.values.reshape(-1, 1).astype(np.float32)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        res_scaled = scaler.fit_transform(res_values)

        X_res, y_res = create_sequences(res_scaled, lookback)

        X_tensor = torch.FloatTensor(X_res)
        y_tensor = torch.FloatTensor(y_res)

        lstm_model = LSTMModel(input_size=1, hidden_size=32, num_layers=1, dropout=0.1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        lstm_model.train()
        for epoch in range(80):
            for bx, by in loader:
                optimizer.zero_grad()
                out = lstm_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Predict residuals for test period
        # Use the last `lookback` residuals to start predicting
        lstm_model.eval()
        last_seq = torch.FloatTensor(res_scaled[-lookback:].reshape(1, lookback, 1))

        lstm_residual_forecast = []
        with torch.no_grad():
            for _ in range(n - train_size):
                pred = lstm_model(last_seq)
                lstm_residual_forecast.append(pred.item())
                # Shift sequence
                new_seq = torch.cat([last_seq[:, 1:, :],
                                     pred.unsqueeze(0).unsqueeze(-1)], dim=1)
                last_seq = new_seq

        lstm_residual_forecast = scaler.inverse_transform(
            np.array(lstm_residual_forecast).reshape(-1, 1)
        ).flatten()

    except Exception as e:
        print(f"    Hybrid LSTM step failed: {e}. Using SARIMA only.")
        lstm_residual_forecast = np.zeros(n - train_size)

    # Step 3: Combine
    hybrid_forecast = sarima_forecast.values + lstm_residual_forecast
    test = series.iloc[train_size:]

    mape = mean_absolute_percentage_error(test, hybrid_forecast) * 100
    rmse = np.sqrt(mean_squared_error(test, hybrid_forecast))

    return {
        "model_type": "Hybrid (SARIMA + LSTM)",
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "sarima_mape": round(mean_absolute_percentage_error(test, sarima_forecast) * 100, 2),
        "actual": test.values.tolist(),
        "predicted": hybrid_forecast.tolist(),
        "sarima_predicted": sarima_forecast.values.tolist(),
        "dates": [str(d) for d in test.index],
        "train_size": train_size,
        "test_size": len(test),
    }


# --- Master Forecasting Pipeline --------------------------------------------

def run_forecasting():
    """Run all forecasting models for each spice."""

    print("=" * 70)
    print("FORECASTING PIPELINE")
    print("=" * 70)

    df_monthly = pd.read_csv(PROCESSED_DIR / "spice_monthly_national.csv", parse_dates=["date"])

    all_results = {}

    for spice in df_monthly["spice"].unique():
        print(f"\n{'-' * 50}")
        print(f"Forecasting: {spice}")
        print(f"{'-' * 50}")

        sdf = df_monthly[df_monthly["spice"] == spice].set_index("date").sort_index()
        series = sdf["modal_price_mean"].dropna()

        if len(series) < 36:
            print(f"  [WARN] Too few months ({len(series)}), skipping")
            continue

        spice_results = {}

        # SARIMA
        print("  [1/3] Fitting SARIMA...")
        sarima_res = fit_sarima(series)
        if sarima_res:
            # Remove non-serializable objects
            fitted_model = sarima_res.pop("fitted_model", None)
            spice_results["sarima"] = sarima_res
            print(f"    MAPE: {sarima_res['mape']:.2f}%")

            # Save SARIMA model
            if fitted_model:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                safe_name = spice.replace(" ", "_").replace("(", "").replace(")", "")
                joblib.dump(fitted_model, MODELS_DIR / f"sarima_{safe_name}.joblib")

        # LSTM
        print("  [2/3] Fitting LSTM...")
        lstm_res = fit_lstm(series)
        if lstm_res:
            # Save model state and remove from JSON
            safe_name = spice.replace(" ", "_").replace("(", "").replace(")", "")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(lstm_res.pop("model_state", {}), MODELS_DIR / f"lstm_{safe_name}.pt")
            joblib.dump(lstm_res.pop("scaler", None), MODELS_DIR / f"scaler_{safe_name}.joblib")
            spice_results["lstm"] = lstm_res
            print(f"    MAPE: {lstm_res['mape']:.2f}%")

        # Hybrid
        print("  [3/3] Fitting Hybrid (SARIMA + LSTM)...")
        hybrid_res = fit_hybrid(series)
        if hybrid_res:
            spice_results["hybrid"] = hybrid_res
            print(f"    MAPE: {hybrid_res['mape']:.2f}%")

        # Comparison
        if spice_results:
            best_model = min(
                [(k, v["mape"]) for k, v in spice_results.items()],
                key=lambda x: x[1]
            )
            spice_results["best_model"] = best_model[0]
            spice_results["best_mape"] = best_model[1]
            print(f"  [OK] Best model: {best_model[0]} (MAPE: {best_model[1]:.2f}%)")

        all_results[spice] = spice_results

    # Save forecast results
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRECOMPUTED_DIR / "forecast_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Forecast results saved to {PRECOMPUTED_DIR / 'forecast_results.json'}")
    return all_results


if __name__ == "__main__":
    run_forecasting()
