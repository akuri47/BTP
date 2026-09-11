# Price Dynamics and Adulteration Forecasting in Spices

**BTP Project** — Data Science Analysis & Forecasting Dashboard for Indian Spices

## Overview

This project investigates whether spice price volatility in Indian wholesale markets (mandis) correlates with and predicts food adulteration incidence. It covers 5 major Indian spices: **Turmeric, Dry Chillies, Cumin (Jeera), Coriander, and Black Pepper**.

## Key Features

- **Price Trend Analysis** with structural break detection
- **GARCH(1,1) Volatility Modeling** per spice
- **Cross-Correlation & Granger Causality** testing (price volatility → adulteration)
- **Hybrid Forecasting** (SARIMA + LSTM) with MAPE evaluation
- **Interactive Streamlit Dashboard** for exploration

## Data Sources

| Source | What | Frequency |
|--------|------|-----------|
| [Agmarknet / data.gov.in](https://data.gov.in) | APMC mandi wholesale prices | Daily (2006–2026) |
| [EU RASFF](https://food.ec.europa.eu/safety/rasff-food-and-feed-safety-alerts/rasff-window_en) | Border rejection notifications for Indian spices | Daily (2000–2026) |
| [US FDA Import Refusals](https://datadashboard.fda.gov/oii/index.htm) | Spice shipment refusals (Industry Code 28) | Monthly (2005–2026) |
| [FSSAI Parliamentary Replies](https://sansad.in) | Domestic spice testing results | Annual (2022–2024) |

## Project Structure

```
BTP/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml        # API key (gitignored)
├── data/
│   ├── raw/                # Raw API responses
│   ├── processed/          # Cleaned, aggregated CSVs
│   └── precomputed/        # Analysis results for dashboard
├── models/                 # Saved SARIMA/LSTM models
├── notebooks/              # Training pipeline notebooks
└── src/
    ├── data_acquisition.py # API fetch + clean
    ├── analysis.py         # ADF, GARCH, Granger causality
    └── forecasting.py      # SARIMA + LSTM + Hybrid
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your API key (if fetching fresh Agmarknet data):
```bash
cp .env.example .env
```

### 3. Run data pipeline (optional - precomputed data included)
```bash
python -m src.data_acquisition
```

### 4. Run analysis (optional - precomputed data included)
```bash
python -m src.analysis
python -m src.forecasting
```

### 5. Launch dashboard
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → Select this repo → Point to `app.py`
4. Add your API key in Streamlit Cloud's Secrets panel:
   ```toml
   [api]
   DATA_GOV_API_KEY = "your-key-here"
   ```
5. Deploy!

> **Note:** The dashboard loads precomputed results only. Heavy model training runs offline via `src/forecasting.py`.

## Methodology

- **Stationarity:** ADF test on levels and first differences
- **Structural Breaks:** CUSUM-based regime shift detection
- **Volatility:** GARCH(1,1) with persistence analysis
- **Correlation:** Pearson cross-correlation at ±12 monthly lags
- **Causality:** Granger causality (F-test, lags 1–6)
- **Forecasting:** SARIMA(1,1,1)(1,1,1,12) + LSTM (2-layer, 64 hidden) → Hybrid ensemble

## License

Data is sourced under [Government Open Data License — India (GODL)](https://data.gov.in/Godl).
