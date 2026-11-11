"""
Statistical Analysis Module — Spice Price Dynamics BTP
=======================================================
ADF stationarity tests, Bai-Perron structural break detection,
GARCH(1,1) volatility modeling, cross-correlation lag analysis,
and Granger causality tests.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from arch import arch_model
from pathlib import Path
import json


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PRECOMPUTED_DIR = Path(__file__).resolve().parent.parent / "data" / "precomputed"


# --- ADF Stationarity Test --------------------------------------------------

def run_adf_tests(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller test on each spice's monthly modal price series.
    Tests both levels and first-differences (log returns).
    """
    results = []

    for spice in df_monthly["spice"].unique():
        series = df_monthly[df_monthly["spice"] == spice].set_index("date")["modal_price_mean"].dropna()

        if len(series) < 20:
            print(f"  [WARN] {spice}: too few observations ({len(series)}), skipping ADF")
            continue

        # Test on levels
        adf_level = adfuller(series, autolag="AIC")
        results.append({
            "spice": spice,
            "test_on": "Level (Price)",
            "adf_statistic": round(adf_level[0], 4),
            "p_value": round(adf_level[1], 4),
            "lags_used": adf_level[2],
            "n_obs": adf_level[3],
            "critical_1pct": round(adf_level[4]["1%"], 4),
            "critical_5pct": round(adf_level[4]["5%"], 4),
            "stationary": bool(adf_level[1] < 0.05),
        })

        # Test on first differences (log returns)
        log_ret = np.log(series / series.shift(1)).dropna()
        if len(log_ret) > 10:
            adf_diff = adfuller(log_ret, autolag="AIC")
            results.append({
                "spice": spice,
                "test_on": "First Diff (Log Return)",
                "adf_statistic": round(adf_diff[0], 4),
                "p_value": round(adf_diff[1], 4),
                "lags_used": adf_diff[2],
                "n_obs": adf_diff[3],
                "critical_1pct": round(adf_diff[4]["1%"], 4),
                "critical_5pct": round(adf_diff[4]["5%"], 4),
                "stationary": bool(adf_diff[1] < 0.05),
            })

    df_results = pd.DataFrame(results)
    print("\n-- ADF Stationarity Test Results --")
    print(df_results.to_string(index=False))
    return df_results


# --- Structural Break Detection ---------------------------------------------

def detect_structural_breaks(df_monthly: pd.DataFrame, min_segment_pct: float = 0.15) -> dict:
    """
    Detect structural break points in each spice's price series using
    a rolling Chow-test inspired approach with CUSUM-based detection.

    Also uses a simpler regime-change detection based on rolling mean shifts.
    """
    all_breaks = {}

    for spice in df_monthly["spice"].unique():
        sdf = df_monthly[df_monthly["spice"] == spice].set_index("date")["modal_price_mean"].dropna()

        if len(sdf) < 24:
            print(f"  [WARN] {spice}: too few observations for break detection")
            all_breaks[spice] = []
            continue

        # Method: Rolling z-score of price changes to detect regime shifts
        log_prices = np.log(sdf)
        returns = log_prices.diff().dropna()

        # Rolling statistics
        window = max(6, len(returns) // 8)  # adaptive window
        roll_mean = returns.rolling(window=window, center=True).mean()
        roll_std = returns.rolling(window=window, center=True).std()

        # Z-score of deviation from rolling mean
        z_scores = (returns - roll_mean) / roll_std
        z_scores = z_scores.dropna()

        # Identify break candidates: months where |z| > 2.0
        # and also where cumulative sum shows persistent shift
        cumsum = returns.cumsum()
        cumsum_detrended = cumsum - np.linspace(cumsum.iloc[0], cumsum.iloc[-1], len(cumsum))

        # Find local extrema in detrended cumsum (Bai-Perron approximation)
        breaks = []
        min_segment = int(len(sdf) * min_segment_pct)

        # Simple peak/trough detection on cumsum
        for i in range(min_segment, len(cumsum_detrended) - min_segment):
            window_left = cumsum_detrended.iloc[i - min_segment:i]
            window_right = cumsum_detrended.iloc[i:i + min_segment]

            val = cumsum_detrended.iloc[i]

            # Check if this is a local extremum (break point)
            if (val == window_left.max() and val > window_right.max()) or \
               (val == window_left.min() and val < window_right.min()):
                break_date = cumsum_detrended.index[i]

                # Compute significance: compare mean price before vs after
                price_before = sdf.loc[sdf.index < break_date].tail(min_segment).mean()
                price_after = sdf.loc[sdf.index >= break_date].head(min_segment).mean()
                pct_change = (price_after - price_before) / price_before * 100

                if abs(pct_change) > 10:  # only report >10% shifts
                    breaks.append({
                        "date": break_date.strftime("%Y-%m-%d"),
                        "price_before": round(price_before, 2),
                        "price_after": round(price_after, 2),
                        "pct_change": round(pct_change, 2),
                        "direction": "up" if pct_change > 0 else "down",
                    })

        # Deduplicate breaks within 3 months of each other
        deduped = []
        for b in sorted(breaks, key=lambda x: x["date"]):
            if not deduped or (pd.Timestamp(b["date"]) - pd.Timestamp(deduped[-1]["date"])).days > 90:
                deduped.append(b)

        all_breaks[spice] = deduped
        print(f"  {spice}: {len(deduped)} structural breaks detected")
        for b in deduped:
            print(f"    -> {b['date']}: {b['direction']} {abs(b['pct_change']):.1f}% "
                  f"(₹{b['price_before']:.0f} -> ₹{b['price_after']:.0f})")

    return all_breaks


# --- GARCH Volatility Modeling -----------------------------------------------

def fit_garch_models(df_monthly: pd.DataFrame) -> dict:
    """
    Fit GARCH(1,1) models on each spice's monthly log return series.
    Returns model summaries, conditional volatility series, and persistence metrics.
    """
    garch_results = {}

    for spice in df_monthly["spice"].unique():
        sdf = df_monthly[df_monthly["spice"] == spice].set_index("date").sort_index()
        returns = sdf["log_return"].dropna() * 100  # scale to percentage

        if len(returns) < 30:
            print(f"  [WARN] {spice}: too few observations ({len(returns)}) for GARCH")
            continue

        try:
            # Fit GARCH(1,1) with constant mean
            model = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
            res = model.fit(disp="off", show_warning=False)

            # Extract parameters
            omega = res.params.get("omega", 0)
            alpha = res.params.get("alpha[1]", 0)
            beta = res.params.get("beta[1]", 0)
            persistence = alpha + beta

            # Conditional volatility series
            cond_vol = res.conditional_volatility

            garch_results[spice] = {
                "omega": round(omega, 6),
                "alpha": round(alpha, 4),
                "beta": round(beta, 4),
                "persistence": round(persistence, 4),
                "log_likelihood": round(res.loglikelihood, 2),
                "aic": round(res.aic, 2),
                "bic": round(res.bic, 2),
                "mean_cond_vol": round(cond_vol.mean(), 4),
                "max_cond_vol": round(cond_vol.max(), 4),
                "cond_volatility": cond_vol.reset_index().rename(
                    columns={"index": "date", 0: "cond_vol"}
                ).to_dict(orient="records") if hasattr(cond_vol, "reset_index") else [],
                "interpretation": (
                    f"{'HIGH' if persistence > 0.9 else 'MODERATE' if persistence > 0.7 else 'LOW'} "
                    f"volatility persistence (α+β={persistence:.3f}). "
                    f"{'Shocks are highly persistent — volatility clustering is strong.' if persistence > 0.9 else ''}"
                    f"{'Moderate shock persistence.' if 0.7 < persistence <= 0.9 else ''}"
                    f"{'Shocks decay quickly.' if persistence <= 0.7 else ''}"
                ),
            }

            print(f"  {spice}: a={alpha:.4f}, b={beta:.4f}, persistence={persistence:.4f} "
                  f"-> {'HIGH' if persistence > 0.9 else 'MODERATE' if persistence > 0.7 else 'LOW'}")

        except Exception as e:
            print(f"  [WARN] {spice} GARCH failed: {e}")

    return garch_results


# --- Cross-Correlation & Granger Causality -----------------------------------

def compute_cross_correlations(df_monthly: pd.DataFrame,
                               df_adulteration: pd.DataFrame,
                               max_lag: int = 12) -> dict:
    """
    Compute cross-correlation between spice price volatility (from GARCH or rolling std)
    and adulteration incidence at various lags.
    """
    results = {}

    for spice in df_monthly["spice"].unique():
        price_data = df_monthly[df_monthly["spice"] == spice].set_index("date").sort_index()
        adult_data = df_adulteration[df_adulteration["spice"] == spice].set_index("date").sort_index()

        if price_data.empty or adult_data.empty:
            continue

        # Use rolling std of log returns as volatility proxy
        vol_series = price_data["log_return"].rolling(window=3).std().dropna()

        # Align series
        common_dates = vol_series.index.intersection(adult_data.index)
        if len(common_dates) < 20:
            print(f"  [WARN] {spice}: insufficient overlapping data ({len(common_dates)} months)")
            continue

        vol_aligned = vol_series.loc[common_dates]
        incident_aligned = adult_data.loc[common_dates, "incident_count"]

        # Cross-correlation at various lags
        lag_correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                # Price volatility LEADS adulteration by `lag` months
                v = vol_aligned.iloc[:len(vol_aligned) - lag]
                a = incident_aligned.iloc[lag:]
            elif lag < 0:
                # Adulteration LEADS price volatility by `|lag|` months
                v = vol_aligned.iloc[-lag:]
                a = incident_aligned.iloc[:len(incident_aligned) + lag]
            else:
                v = vol_aligned
                a = incident_aligned

            if len(v) > 5 and len(a) > 5:
                corr, pval = stats.pearsonr(v.values, a.values[:len(v)])
                lag_correlations.append({
                    "lag": lag,
                    "correlation": round(corr, 4),
                    "p_value": round(pval, 4),
                    "significant": bool(pval < 0.05),
                })

        # Find the lag with strongest positive correlation
        pos_corrs = [lc for lc in lag_correlations if lc["correlation"] > 0]
        best_lag = max(pos_corrs, key=lambda x: x["correlation"]) if pos_corrs else None

        results[spice] = {
            "lag_correlations": lag_correlations,
            "best_lag": best_lag,
            "n_observations": len(common_dates),
        }

        if best_lag:
            print(f"  {spice}: best lag = {best_lag['lag']} months, "
                  f"r = {best_lag['correlation']:.3f}, p = {best_lag['p_value']:.3f}"
                  f"{' *' if best_lag['significant'] else ''}")

    return results


def run_granger_causality(df_monthly: pd.DataFrame,
                          df_adulteration: pd.DataFrame,
                          max_lag: int = 6) -> dict:
    """
    Run Granger causality tests: does price volatility Granger-cause adulteration?
    """
    results = {}

    for spice in df_monthly["spice"].unique():
        price_data = df_monthly[df_monthly["spice"] == spice].set_index("date").sort_index()
        adult_data = df_adulteration[df_adulteration["spice"] == spice].set_index("date").sort_index()

        if price_data.empty or adult_data.empty:
            continue

        # Volatility proxy: rolling 3-month std of log returns
        vol_series = price_data["log_return"].rolling(window=3).std().dropna()

        # Align
        common_dates = vol_series.index.intersection(adult_data.index)
        if len(common_dates) < 30:
            print(f"  [WARN] {spice}: insufficient data for Granger test ({len(common_dates)} months)")
            continue

        vol_aligned = vol_series.loc[common_dates].values
        incident_aligned = adult_data.loc[common_dates, "incident_count"].values

        # Combine into a 2-column array
        combined = np.column_stack([incident_aligned[:len(vol_aligned)], vol_aligned])

        # Remove NaNs
        mask = ~np.isnan(combined).any(axis=1)
        combined = combined[mask]

        if len(combined) < max_lag + 10:
            continue

        try:
            gc_result = grangercausalitytests(combined, maxlag=max_lag)

            lag_results = []
            for lag in range(1, max_lag + 1):
                if lag in gc_result:
                    test_stats = gc_result[lag][0]
                    f_stat = test_stats["ssr_ftest"][0]
                    p_val = test_stats["ssr_ftest"][1]
                    lag_results.append({
                        "lag": lag,
                        "f_statistic": round(f_stat, 4),
                        "p_value": round(p_val, 4),
                        "significant": bool(p_val < 0.05),
                    })

            # Best (most significant) lag
            sig_results = [lr for lr in lag_results if lr["significant"]]

            results[spice] = {
                "lag_results": lag_results,
                "any_significant": len(sig_results) > 0,
                "best_lag": min(sig_results, key=lambda x: x["p_value"]) if sig_results else None,
                "n_observations": len(combined),
            }

            sig_str = f"{len(sig_results)}/{len(lag_results)} lags significant" if lag_results else "N/A"
            print(f"  {spice}: Granger causality — {sig_str}")

        except Exception as e:
            print(f"  [WARN] {spice} Granger test failed: {e}")

    return results


# --- Master Analysis Pipeline -----------------------------------------------

def run_full_analysis():
    """Load processed data and run all statistical analyses."""

    print("=" * 70)
    print("STATISTICAL ANALYSIS PIPELINE")
    print("=" * 70)

    # Load data
    df_monthly = pd.read_csv(PROCESSED_DIR / "spice_monthly_national.csv", parse_dates=["date"])
    df_adult = pd.read_csv(PROCESSED_DIR / "adulteration_incidents.csv", parse_dates=["date"])

    print(f"\nLoaded: {len(df_monthly)} monthly price rows, {len(df_adult)} adulteration rows")
    print(f"Spices: {df_monthly['spice'].unique().tolist()}")

    # 1. ADF Tests
    print("\n" + "=" * 50)
    print("[1/5] ADF STATIONARITY TESTS")
    print("=" * 50)
    adf_results = run_adf_tests(df_monthly)

    # 2. Structural Breaks
    print("\n" + "=" * 50)
    print("[2/5] STRUCTURAL BREAK DETECTION")
    print("=" * 50)
    breaks = detect_structural_breaks(df_monthly)

    # 3. GARCH
    print("\n" + "=" * 50)
    print("[3/5] GARCH(1,1) VOLATILITY MODELS")
    print("=" * 50)
    garch_results = fit_garch_models(df_monthly)

    # 4. Cross-Correlation
    print("\n" + "=" * 50)
    print("[4/5] CROSS-CORRELATION LAG ANALYSIS")
    print("=" * 50)
    xcorr_results = compute_cross_correlations(df_monthly, df_adult)

    # 5. Granger Causality
    print("\n" + "=" * 50)
    print("[5/5] GRANGER CAUSALITY TESTS")
    print("=" * 50)
    granger_results = run_granger_causality(df_monthly, df_adult)

    # -- Save all results --
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

    adf_results.to_csv(PRECOMPUTED_DIR / "adf_results.csv", index=False)

    # Save breaks
    with open(PRECOMPUTED_DIR / "structural_breaks.json", "w") as f:
        json.dump(breaks, f, indent=2, default=str)

    # Save GARCH (without the full volatility series in JSON, save series separately)
    garch_summary = {}
    for spice, res in garch_results.items():
        # Save conditional volatility series
        if res.get("cond_volatility"):
            vol_df = pd.DataFrame(res["cond_volatility"])
            vol_df.to_csv(PRECOMPUTED_DIR / f"garch_vol_{spice.replace(' ', '_').replace('(', '').replace(')', '')}.csv", index=False)

        garch_summary[spice] = {k: v for k, v in res.items() if k != "cond_volatility"}

    with open(PRECOMPUTED_DIR / "garch_results.json", "w") as f:
        json.dump(garch_summary, f, indent=2)

    # Save cross-correlations
    xcorr_save = {}
    for spice, res in xcorr_results.items():
        xcorr_save[spice] = {
            "lag_correlations": res["lag_correlations"],
            "best_lag": res["best_lag"],
            "n_observations": res["n_observations"],
        }
    with open(PRECOMPUTED_DIR / "cross_correlations.json", "w") as f:
        json.dump(xcorr_save, f, indent=2)

    # Save Granger
    with open(PRECOMPUTED_DIR / "granger_causality.json", "w") as f:
        json.dump(granger_results, f, indent=2)

    print(f"\n[OK] All analysis results saved to {PRECOMPUTED_DIR}")
    return adf_results, breaks, garch_results, xcorr_results, granger_results


if __name__ == "__main__":
    run_full_analysis()
