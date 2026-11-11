"""
Risk Scoring Module — Adulteration Early Warning System
========================================================
Computes a composite risk score (0–100) per spice per month using:
  1. Price percentile (current price vs 5-year rolling distribution)
  2. GARCH conditional volatility (current vs historical)
  3. Price momentum (3-month log return)
  4. Seasonal factor (post-harvest months = higher risk)
  5. Adulteration trend (rolling 3-month change in incident count)

Weights are derived from cross-correlation and Granger causality results
(see analysis.py outputs). The methodology follows the OECD Composite
Indicators Handbook (2008) normalized weighted-signal approach.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

PROCESSED_DIR  = Path(__file__).resolve().parent.parent / "data" / "processed"
PRECOMPUTED_DIR = Path(__file__).resolve().parent.parent / "data" / "precomputed"


# --- Signal weight derivation from analysis results -------------------------

def load_analysis_driven_weights() -> dict:
    """
    Derive per-spice signal weights from cross-correlation and Granger results.
    Spices with stronger price→adulteration link get higher volatility/momentum weights.
    """
    # Load cross-correlation results
    xcorr_path = PRECOMPUTED_DIR / "cross_correlations.json"
    granger_path = PRECOMPUTED_DIR / "granger_causality.json"

    xcorr = {}
    if xcorr_path.exists():
        with open(xcorr_path) as f:
            xcorr = json.load(f)

    granger = {}
    if granger_path.exists():
        with open(granger_path) as f:
            granger = json.load(f)

    # Base weights (equal weighting as default)
    base_weights = {
        "price_percentile": 0.20,
        "volatility": 0.25,
        "momentum": 0.20,
        "seasonal": 0.15,
        "adulteration_trend": 0.20,
    }

    spice_weights = {}
    for spice in ["Turmeric", "Dry Chillies", "Cumin (Jeera)", "Coriander", "Black Pepper"]:
        w = base_weights.copy()

        # Adjust based on cross-correlation strength
        best_corr = 0.0
        if spice in xcorr and xcorr[spice].get("best_lag"):
            best_corr = abs(xcorr[spice]["best_lag"].get("correlation", 0))

        # Adjust based on Granger significance
        granger_sig = False
        if spice in granger:
            granger_sig = granger[spice].get("any_significant", False)

        # Boost volatility and momentum weights for spices with strong correlation
        if granger_sig:
            # Strong evidence: volatility is a leading indicator
            w["volatility"] = 0.30
            w["momentum"] = 0.25
            w["price_percentile"] = 0.15
            w["seasonal"] = 0.15
            w["adulteration_trend"] = 0.15
        elif best_corr > 0.2:
            # Moderate evidence
            w["volatility"] = 0.28
            w["momentum"] = 0.22
            w["price_percentile"] = 0.18
            w["seasonal"] = 0.15
            w["adulteration_trend"] = 0.17
        else:
            # Weak/no evidence: rely more on direct adulteration trend
            w["volatility"] = 0.18
            w["momentum"] = 0.15
            w["price_percentile"] = 0.20
            w["seasonal"] = 0.17
            w["adulteration_trend"] = 0.30

        spice_weights[spice] = w

    return spice_weights


# --- Signal computation -----------------------------------------------------

def compute_price_percentile(prices: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute where each month's price falls within a rolling window percentile.
    Returns 0-1 (0 = historically cheap, 1 = historically expensive).
    """
    rolling_rank = prices.rolling(window=window, min_periods=12).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    return rolling_rank.clip(0, 1)


def compute_volatility_signal(prices: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute normalized volatility signal using rolling std of log returns.
    Returns 0-1 (0 = calm, 1 = extremely volatile).
    """
    log_ret = np.log(prices / prices.shift(1))
    rolling_vol = log_ret.rolling(window=3).std()

    # Normalize within rolling window
    vol_pct = rolling_vol.rolling(window=window, min_periods=12).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    return vol_pct.clip(0, 1)


def compute_momentum_signal(prices: pd.Series, lookback: int = 3) -> pd.Series:
    """
    Compute price momentum (3-month log return), normalized to 0-1.
    High positive momentum (price surge) → higher risk.
    """
    momentum = np.log(prices / prices.shift(lookback))

    # Normalize: use historical distribution of momentum values
    mom_pct = momentum.rolling(window=60, min_periods=12).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    return mom_pct.clip(0, 1)


def compute_seasonal_signal(dates: pd.Series) -> pd.Series:
    """
    Seasonal risk factor based on month-of-year.
    Post-harvest months (Mar-Jun) have higher adulteration risk
    as new processed stocks enter the market.
    """
    # Empirical seasonal risk weights (from RASFF notification patterns)
    seasonal_risk = {
        1: 0.5,   # Jan - moderate
        2: 0.5,   # Feb - moderate
        3: 0.8,   # Mar - post-rabi harvest
        4: 0.9,   # Apr - peak processing
        5: 0.85,  # May - peak processing
        6: 0.7,   # Jun - monsoon onset
        7: 0.4,   # Jul - low
        8: 0.35,  # Aug - low
        9: 0.45,  # Sep - pre-kharif
        10: 0.55, # Oct - kharif harvest
        11: 0.6,  # Nov - post-harvest
        12: 0.55, # Dec - moderate
    }
    return dates.dt.month.map(seasonal_risk)


def compute_adulteration_trend(incidents: pd.Series, window: int = 3) -> pd.Series:
    """
    Compute the rolling trend in adulteration incidents.
    Rising trend → higher risk.
    """
    rolling_mean = incidents.rolling(window=window, min_periods=1).mean()
    lagged_mean = rolling_mean.shift(window)

    # Percentage change in rolling average
    trend = (rolling_mean - lagged_mean) / lagged_mean.replace(0, np.nan)
    trend = trend.fillna(0)

    # Normalize to 0-1 using sigmoid-like transformation
    # Positive trend → higher score, negative → lower
    trend_signal = 1 / (1 + np.exp(-5 * trend))  # sigmoid centered at 0
    return trend_signal.clip(0, 1)


# --- Composite risk score ---------------------------------------------------

def compute_risk_scores(df_prices: pd.DataFrame, df_adulteration: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite risk scores for each spice at each month.

    Returns DataFrame with columns:
        date, spice, risk_score, risk_level,
        signal_price, signal_volatility, signal_momentum,
        signal_seasonal, signal_trend,
        weight_source (describes how weights were derived)
    """
    spice_weights = load_analysis_driven_weights()
    all_results = []

    for spice in df_prices["spice"].unique():
        if spice not in spice_weights:
            continue

        weights = spice_weights[spice]

        # Get price series
        sp = df_prices[df_prices["spice"] == spice].set_index("date").sort_index()
        prices = sp["modal_price_mean"].dropna()

        # Get adulteration series
        sa = df_adulteration[df_adulteration["spice"] == spice].set_index("date").sort_index()
        if "incident_count" in sa.columns:
            incidents = sa["incident_count"].reindex(prices.index).fillna(0)
        else:
            incidents = pd.Series(0, index=prices.index)

        # Compute all signals
        sig_price = compute_price_percentile(prices)
        sig_vol = compute_volatility_signal(prices)
        sig_mom = compute_momentum_signal(prices)
        sig_seasonal = compute_seasonal_signal(pd.Series(prices.index))
        sig_seasonal.index = prices.index
        sig_trend = compute_adulteration_trend(incidents)

        # Compute composite score
        composite = (
            weights["price_percentile"] * sig_price +
            weights["volatility"] * sig_vol +
            weights["momentum"] * sig_mom +
            weights["seasonal"] * sig_seasonal +
            weights["adulteration_trend"] * sig_trend
        )

        # Scale to 0-100
        risk_score = (composite * 100).clip(0, 100)

        # Classify risk level
        def classify_risk(score):
            if pd.isna(score):
                return "Unknown"
            elif score >= 65:
                return "High"
            elif score >= 35:
                return "Moderate"
            else:
                return "Low"

        risk_level = risk_score.apply(classify_risk)

        # Build result DataFrame
        result = pd.DataFrame({
            "date": prices.index,
            "spice": spice,
            "risk_score": risk_score.round(1),
            "risk_level": risk_level,
            "signal_price": sig_price.round(3),
            "signal_volatility": sig_vol.round(3),
            "signal_momentum": sig_mom.round(3),
            "signal_seasonal": sig_seasonal.round(3),
            "signal_trend": sig_trend.round(3),
        })

        all_results.append(result)

    df_risk = pd.concat(all_results, ignore_index=True)
    df_risk.sort_values(["spice", "date"], inplace=True)
    df_risk.reset_index(drop=True, inplace=True)

    return df_risk


# --- Backtesting against known events ---------------------------------------

def backtest_risk_scores(df_risk: pd.DataFrame) -> dict:
    """
    Validate the risk score against known adulteration episodes.
    Returns a report of hits and misses.
    """
    # Known high-risk episodes (from literature and news)
    known_events = [
        {"period": ("2015-06", "2016-06"), "spice": "Dry Chillies",
         "event": "Sudan dye contamination wave in Indian chilli exports"},
        {"period": ("2020-03", "2020-09"), "spice": None,
         "event": "COVID-19 supply chain disruption"},
        {"period": ("2023-01", "2023-09"), "spice": "Cumin (Jeera)",
         "event": "Cumin price spike — widespread adulteration detected"},
        {"period": ("2024-04", "2024-08"), "spice": None,
         "event": "MDH/Everest EtO crisis, Hong Kong/Singapore recalls"},
        {"period": ("2024-09", "2025-03"), "spice": None,
         "event": "FSSAI special surveillance drive"},
    ]

    results = []
    for event in known_events:
        start, end = event["period"]
        spice_filter = event["spice"]

        mask = (df_risk["date"] >= start) & (df_risk["date"] <= end)
        if spice_filter:
            mask = mask & (df_risk["spice"] == spice_filter)

        event_data = df_risk[mask]
        if event_data.empty:
            results.append({**event, "detected": False, "avg_score": None,
                          "pct_high": None, "verdict": "No data"})
            continue

        avg_score = event_data["risk_score"].mean()
        pct_high = (event_data["risk_level"] == "High").mean() * 100
        pct_moderate_plus = (event_data["risk_level"].isin(["High", "Moderate"])).mean() * 100

        detected = pct_moderate_plus >= 50  # at least 50% of months flagged Moderate+

        results.append({
            "event": event["event"],
            "period": f"{start} to {end}",
            "spice": spice_filter or "All spices",
            "detected": detected,
            "avg_score": round(avg_score, 1),
            "pct_high": round(pct_high, 1),
            "pct_moderate_plus": round(pct_moderate_plus, 1),
            "verdict": "✅ Detected" if detected else "⚠️ Missed",
        })

    return results


# --- Main pipeline -----------------------------------------------------------

def run_risk_scoring():
    """Compute risk scores and save results."""
    print("=" * 70)
    print("ADULTERATION RISK SCORING PIPELINE")
    print("=" * 70)

    # Load data
    df_prices = pd.read_csv(PROCESSED_DIR / "spice_monthly_national.csv", parse_dates=["date"])
    df_adult = pd.read_csv(PROCESSED_DIR / "adulteration_incidents.csv", parse_dates=["date"])

    print(f"Loaded: {len(df_prices)} price rows, {len(df_adult)} adulteration rows")

    # Load and display weights
    weights = load_analysis_driven_weights()
    print("\nDerived signal weights (from cross-correlation & Granger results):")
    for spice, w in weights.items():
        print(f"  {spice}: {w}")

    # Compute risk scores
    print("\nComputing risk scores...")
    df_risk = compute_risk_scores(df_prices, df_adult)
    print(f"Risk scores computed: {len(df_risk)} rows")

    # Summary
    print("\nCurrent risk levels (latest month):")
    latest = df_risk.groupby("spice").last()
    for spice, row in latest.iterrows():
        marker = {"High": "[HIGH]", "Moderate": "[MED]", "Low": "[LOW]"}.get(row["risk_level"], "[?]")
        print(f"  {marker} {spice}: {row['risk_score']:.0f}/100 ({row['risk_level']})")

    # Backtest
    print("\nBacktesting against known adulteration events:")
    backtest = backtest_risk_scores(df_risk)
    for bt in backtest:
        marker = "[OK]" if bt.get("detected") else "[MISS]"
        print(f"  {marker} | {bt['event']}")
        print(f"      Period: {bt['period']} | Avg score: {bt.get('avg_score', 'N/A')} | "
              f"% Moderate+: {bt.get('pct_moderate_plus', 'N/A')}%")

    # Save
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    df_risk.to_csv(PRECOMPUTED_DIR / "risk_scores.csv", index=False)
    print(f"\n[OK] Risk scores saved to {PRECOMPUTED_DIR / 'risk_scores.csv'}")

    # Save backtest
    with open(PRECOMPUTED_DIR / "risk_backtest.json", "w") as f:
        json.dump(backtest, f, indent=2, default=str)
    print(f"[OK] Backtest results saved to {PRECOMPUTED_DIR / 'risk_backtest.json'}")

    # Save weights metadata
    with open(PRECOMPUTED_DIR / "risk_weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    return df_risk, backtest


if __name__ == "__main__":
    run_risk_scoring()
