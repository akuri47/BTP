# Price Dynamics and Adulteration Forecasting in Spices

**BTP Project** — Data Science Analysis & Forecasting Dashboard for Indian Spices

## Overview

This project investigates whether spice price volatility in Indian wholesale markets (mandis) correlates with and predicts food adulteration incidence. It covers 5 major Indian spices: **Turmeric, Dry Chillies, Cumin (Jeera), Coriander, and Black Pepper**.

## Key Features

- **Price Trend Analysis** with structural break detection
- **GARCH(1,1) Volatility Modeling** per spice
- **Cross-Correlation & Granger Causality** testing (price volatility → adulteration)
- **Hybrid Forecasting** (SARIMA + LSTM) with MAPE evaluation
- **Adulteration Risk Early Warning System** — composite risk score (0–100) with traffic-light classification, validated against 5 known events
- **Multi-Source Adulteration Data** — openFDA (scraped) + EU RASFF (published) + FSSAI (Parliamentary replies)
- **Interactive Streamlit Dashboard** for exploration

## Data Sources

| Source | What | Frequency | Access |
|--------|------|-----------|--------|
| [Agmarknet / data.gov.in](https://data.gov.in) | APMC mandi wholesale prices | Daily (2014–2026) | API (real-time) |
| [openFDA API](https://api.fda.gov) | US FDA food enforcement actions (recalls, seizures) for spice products | Event-based | API (scraped) |
| [EU RASFF Annual Reports](https://food.ec.europa.eu/safety/rasff-food-and-feed-safety-alerts_en) | Border rejection notifications for India-origin herbs & spices | Annual aggregates | Published reports |
| [FSSAI Parliamentary Replies](https://sansad.in) | Domestic spice testing results & failure rates | Annual (exact FY2022-24) | Rajya Sabha Q.1907, Lok Sabha Q.1252 |

## Project Structure

```
BTP/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml        # API key (gitignored)
├── data/
│   ├── raw/                # Raw API responses + openFDA JSON
│   ├── processed/          # Cleaned CSVs (prices + multi-source adulteration)
│   └── precomputed/        # Analysis results for dashboard
├── models/                 # Saved SARIMA/LSTM models
├── notebooks/              # Training pipeline notebooks
└── src/
    ├── data_acquisition.py # API fetch + clean + multi-source adulteration
    ├── analysis.py         # ADF, GARCH, Granger causality
    ├── forecasting.py      # SARIMA + LSTM + Hybrid
    └── risk_scoring.py     # Composite risk index + backtesting
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
- **Risk Scoring:** OECD-style composite index using 5 normalized signals (price percentile, volatility, momentum, seasonality, adulteration trend) with analysis-derived weights; backtested against 5 known events (5/5 detected)

## License

Data is sourced under [Government Open Data License — India (GODL)](https://data.gov.in/Godl).
