"""
Streamlit Dashboard — Spice Price Dynamics & Adulteration Correlation
=====================================================================
Loads precomputed results (no heavy training at runtime).
Deploy via: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spice Price Dynamics & Adulteration Correlation — India",
    page_icon="SP",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Data Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"


# ─── Load Data (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_monthly_prices():
    return pd.read_csv(PROCESSED_DIR / "spice_monthly_national.csv", parse_dates=["date"])

@st.cache_data
def load_statewise_prices():
    path = PROCESSED_DIR / "spice_monthly_statewise.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None

@st.cache_data
def load_adulteration():
    return pd.read_csv(PROCESSED_DIR / "adulteration_incidents.csv", parse_dates=["date"])

@st.cache_data
def load_adf_results():
    path = PRECOMPUTED_DIR / "adf_results.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_structural_breaks():
    path = PRECOMPUTED_DIR / "structural_breaks.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_garch_results():
    path = PRECOMPUTED_DIR / "garch_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_cross_correlations():
    path = PRECOMPUTED_DIR / "cross_correlations.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_granger_results():
    path = PRECOMPUTED_DIR / "granger_causality.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_forecast_results():
    path = PRECOMPUTED_DIR / "forecast_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_garch_vol(spice):
    safe = spice.replace(" ", "_").replace("(", "").replace(")", "")
    path = PRECOMPUTED_DIR / f"garch_vol_{safe}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None

@st.cache_data
def load_risk_scores():
    path = PRECOMPUTED_DIR / "risk_scores.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None

@st.cache_data
def load_risk_backtest():
    path = PRECOMPUTED_DIR / "risk_backtest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

@st.cache_data
def load_risk_weights():
    path = PRECOMPUTED_DIR / "risk_weights.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("Spice Analytics")
st.sidebar.markdown("**Spice Price Dynamics & Adulteration-Incidence Correlation in India**")
st.sidebar.markdown("---")

df_monthly = load_monthly_prices()
spices = sorted(df_monthly["spice"].unique().tolist())

selected_spice = st.sidebar.selectbox("Select Spice", spices, index=0)

# Date range
min_date = df_monthly["date"].min().date()
max_date = df_monthly["date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# View toggle
view_mode = st.sidebar.radio("View Mode", ["National Average", "State-wise"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Analysis Panels**")
show_risk_dashboard = st.sidebar.checkbox("Risk Early Warning", value=True)
show_price_trend = st.sidebar.checkbox("Price Trend + Break Points", value=True)
show_volatility = st.sidebar.checkbox("GARCH Volatility", value=True)
show_correlation = st.sidebar.checkbox("Cross-Correlation & Granger", value=True)
show_forecast = st.sidebar.checkbox("Forecast (Actual vs Predicted)", value=True)
show_summary = st.sidebar.checkbox("Summary & Insights", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Data: Agmarknet (data.gov.in) | openFDA | EU RASFF | FSSAI")



# ─── Filter Data ─────────────────────────────────────────────────────────────

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

df_spice = df_monthly[
    (df_monthly["spice"] == selected_spice) &
    (df_monthly["date"] >= start_date) &
    (df_monthly["date"] <= end_date)
].copy()

df_adult = load_adulteration()
df_adult_spice = df_adult[
    (df_adult["spice"] == selected_spice) &
    (df_adult["date"] >= start_date) &
    (df_adult["date"] <= end_date)
].copy()


# ─── Title ───────────────────────────────────────────────────────────────────

st.title(f"{selected_spice} — Price Dynamics & Adulteration Analysis")
st.markdown(f"**{start_date.strftime('%b %Y')} — {end_date.strftime('%b %Y')}** | "
            f"{len(df_spice)} monthly observations")


# ─── Panel 0: Risk Early Warning Dashboard ───────────────────────────────────

if show_risk_dashboard:
    st.header("Adulteration Risk — Early Warning System")

    df_risk = load_risk_scores()

    if df_risk is not None and not df_risk.empty:
        # --- Traffic light cards for ALL spices (latest month) ---
        latest_all = df_risk.sort_values("date").groupby("spice").last().reset_index()

        st.subheader("Current Risk Levels (All Spices)")
        cols = st.columns(len(latest_all))
        for i, (_, row) in enumerate(latest_all.iterrows()):
            level = row["risk_level"]
            score = row["risk_score"]
            color_map = {"High": "<span style='color:#ef4444'>&#9679;</span>", "Moderate": "<span style='color:#eab308'>&#9679;</span>", "Low": "<span style='color:#22c55e'>&#9679;</span>", "Unknown": "<span style='color:#9ca3af'>&#9679;</span>"}
            bg_map = {"High": "#fee2e2", "Moderate": "#fef9c3", "Low": "#dcfce7", "Unknown": "#f3f4f6"}

            with cols[i]:
                st.markdown(
                    f"""<div style="background:{bg_map.get(level, '#f3f4f6')};
                    border-radius:12px; padding:16px; text-align:center; border:1px solid #e5e7eb;">
                    <div style="font-size:28px">{color_map.get(level, '<span style=\'color:#9ca3af\'>&#9679;</span>')}</div>
                    <div style="font-size:14px; font-weight:600; margin:4px 0; color:#1f2937">{row['spice']}</div>
                    <div style="font-size:24px; font-weight:700; color:#1f2937">{score:.0f}</div>
                    <div style="font-size:12px; color:#6b7280">{level} Risk</div>
                    </div>""",
                    unsafe_allow_html=True
                )

        st.markdown("")  # spacer

        # --- Selected spice: detailed risk timeline ---
        df_risk_spice = df_risk[
            (df_risk["spice"] == selected_spice) &
            (df_risk["date"] >= start_date) &
            (df_risk["date"] <= end_date)
        ].copy()

        if not df_risk_spice.empty:
            st.subheader(f"{selected_spice} — Risk Score Timeline")

            # Risk score line chart with colored zones
            fig_risk = go.Figure()

            # Background zones
            fig_risk.add_hrect(y0=0, y1=35, fillcolor="rgba(34,197,94,0.08)",
                              line_width=0, annotation_text="Low Risk Zone",
                              annotation_position="bottom left",
                              annotation=dict(font_size=10, font_color="#16a34a"))
            fig_risk.add_hrect(y0=35, y1=65, fillcolor="rgba(234,179,8,0.08)",
                              line_width=0, annotation_text="Moderate",
                              annotation_position="bottom left",
                              annotation=dict(font_size=10, font_color="#ca8a04"))
            fig_risk.add_hrect(y0=65, y1=100, fillcolor="rgba(239,68,68,0.08)",
                              line_width=0, annotation_text="High Risk Zone",
                              annotation_position="bottom left",
                              annotation=dict(font_size=10, font_color="#dc2626"))

            # Color each segment by risk level
            colors = df_risk_spice["risk_level"].map(
                {"High": "#ef4444", "Moderate": "#eab308", "Low": "#22c55e"}
            ).fillna("#9ca3af")

            fig_risk.add_trace(go.Scatter(
                x=df_risk_spice["date"], y=df_risk_spice["risk_score"],
                mode="lines+markers",
                name="Risk Score",
                line=dict(color="#4b5563", width=2),
                marker=dict(color=colors.tolist(), size=6, line=dict(width=1, color="white")),
                hovertemplate="Date: %{x|%b %Y}<br>Risk: %{y:.0f}/100<extra></extra>",
            ))

            fig_risk.update_layout(
                yaxis_title="Risk Score (0–100)", xaxis_title="Date",
                yaxis=dict(range=[0, 100]),
                height=350, hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # --- Signal decomposition (stacked area or radar) ---
            with st.expander("Signal Decomposition — What's Driving the Risk?"):
                signal_cols = {
                    "signal_price": "Price Level",
                    "signal_volatility": "Volatility",
                    "signal_momentum": "Momentum",
                    "signal_seasonal": "Seasonal",
                    "signal_trend": "Adulter. Trend",
                }
                available_signals = [c for c in signal_cols if c in df_risk_spice.columns]

                if available_signals:
                    fig_signals = go.Figure()
                    signal_colors = ["#3b82f6", "#ef4444", "#f97316", "#8b5cf6", "#10b981"]
                    for idx, col in enumerate(available_signals):
                        fig_signals.add_trace(go.Scatter(
                            x=df_risk_spice["date"],
                            y=df_risk_spice[col],
                            mode="lines",
                            name=signal_cols[col],
                            line=dict(width=1.5, color=signal_colors[idx % len(signal_colors)]),
                            stackgroup="one" if idx > 0 else None,
                        ))
                    fig_signals.update_layout(
                        yaxis_title="Signal Strength (0–1)", height=300,
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_signals, use_container_width=True)

                # Latest month signal breakdown
                latest_row = df_risk_spice.iloc[-1]
                st.markdown(f"**Latest month ({latest_row['date'].strftime('%b %Y')}) signal values:**")
                sig_data = []
                weights = load_risk_weights().get(selected_spice, {})
                for col in available_signals:
                    val = latest_row[col]
                    weight = weights.get(col.replace("signal_", ""), 0)
                    contribution = val * weight * 100
                    sig_data.append({
                        "Signal": signal_cols[col],
                        "Value": f"{val:.2f}",
                        "Weight": f"{weight:.0%}",
                        "Contribution": f"{contribution:.1f} pts",
                    })
                st.dataframe(pd.DataFrame(sig_data), use_container_width=True, hide_index=True)

        # --- Backtest results ---
        with st.expander("Backtest — Did it catch known events?"):
            backtest = load_risk_backtest()
            if backtest:
                bt_df = pd.DataFrame(backtest)
                display_cols = ["event", "period", "spice", "avg_score", "pct_moderate_plus", "verdict"]
                available_bt_cols = [c for c in display_cols if c in bt_df.columns]
                st.dataframe(bt_df[available_bt_cols], use_container_width=True, hide_index=True)

                n_detected = sum(1 for b in backtest if b.get("detected"))
                st.success(f"**{n_detected}/{len(backtest)} known adulteration events correctly detected** "
                           f"by the early warning system (≥50% months flagged Moderate or above).")
            else:
                st.info("No backtest results available.")

        # --- Methodology note ---
        with st.expander("Risk Score Methodology"):
            st.markdown("""
            **Composite Risk Index** following the OECD Composite Indicators Handbook (2008):

            1. **Five signals** are computed per spice per month:
               - *Price Percentile* — where current price sits in its 5-year distribution
               - *Volatility* — rolling 3-month std of log returns, normalized historically
               - *Momentum* — 3-month price change, normalized
               - *Seasonal Factor* — empirical monthly risk weights (post-harvest = higher)
               - *Adulteration Trend* — sigmoid-transformed rolling change in incident count

            2. **Normalization:** Each signal is scaled to [0, 1] using rolling percentile rank.

            3. **Weighting:** Weights are **derived from this project's own analysis results**:
               - Spices with significant Granger causality (Turmeric) get higher volatility weights
               - Spices with strong cross-correlation (Dry Chillies, Cumin) get moderately boosted weights
               - Spices with weak correlation (Black Pepper, Coriander) rely more on direct trend signals

            4. **Classification:** Score 0–35 = Low, 35–65 = Moderate, 65–100 = High

            5. **Validation:** Backtested against 5 known adulteration episodes (2015–2025).
            """)

    else:
        st.warning("Risk scores not available. Run `python -m src.risk_scoring` to generate.")


# ─── Panel 1: Price Trend with Structural Breaks ────────────────────────────

if show_price_trend and not df_spice.empty:
    st.header("Price Trend with Structural Break Points")

    breaks = load_structural_breaks().get(selected_spice, [])

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df_spice["date"], y=df_spice["modal_price_mean"],
        mode="lines", name="Modal Price (₹/quintal)",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Date: %{x|%b %Y}<br>Price: ₹%{y:,.0f}/q<extra></extra>"
    ))

    # Min-Max band
    fig.add_trace(go.Scatter(
        x=df_spice["date"], y=df_spice["max_price_mean"],
        mode="lines", name="Max Price", line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df_spice["date"], y=df_spice["min_price_mean"],
        mode="lines", name="Price Range",
        fill="tonexty", fillcolor="rgba(31,119,180,0.1)",
        line=dict(width=0),
    ))

    # Structural break points
    for b in breaks:
        b_date = pd.Timestamp(b["date"])
        if start_date <= b_date <= end_date:
            color = "red" if b["direction"] == "up" else "green"
            fig.add_vline(x=b_date, line_dash="dash", line_color=color, line_width=1.5)
            fig.add_annotation(
                x=b_date, y=df_spice["modal_price_mean"].max() * 0.95,
                text=f"{'↑' if b['direction'] == 'up' else '↓'} {abs(b['pct_change']):.0f}%",
                showarrow=True, arrowhead=2, ax=0, ay=-30,
                font=dict(size=10, color=color),
            )

    fig.update_layout(
        yaxis_title="Price (₹/Quintal)",
        xaxis_title="Date",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # State-wise view
    if view_mode == "State-wise":
        df_state = load_statewise_prices()
        if df_state is not None:
            df_state_spice = df_state[
                (df_state["spice"] == selected_spice) &
                (df_state["date"] >= start_date) &
                (df_state["date"] <= end_date)
            ]
            if not df_state_spice.empty:
                top_states = df_state_spice.groupby("state")["n_observations"].sum().nlargest(8).index
                df_top = df_state_spice[df_state_spice["state"].isin(top_states)]

                fig_state = px.line(df_top, x="date", y="modal_price_mean",
                                    color="state", title="State-wise Price Trends (Top 8 States)")
                fig_state.update_layout(height=350, yaxis_title="Price (₹/Quintal)")
                st.plotly_chart(fig_state, use_container_width=True)

    # Break points table
    if breaks:
        with st.expander(f"Structural Break Points ({len(breaks)} detected)"):
            bp_df = pd.DataFrame(breaks)
            st.dataframe(bp_df, use_container_width=True)


# ─── Panel 2: GARCH Volatility ──────────────────────────────────────────────

if show_volatility:
    st.header("GARCH(1,1) Volatility Analysis")

    garch = load_garch_results().get(selected_spice, {})
    vol_df = load_garch_vol(selected_spice)

    if garch:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("α (ARCH)", f"{garch.get('alpha', 'N/A')}")
        col2.metric("β (GARCH)", f"{garch.get('beta', 'N/A')}")
        col3.metric("Persistence (α+β)", f"{garch.get('persistence', 'N/A')}")
        col4.metric("Mean Cond. Vol.", f"{garch.get('mean_cond_vol', 'N/A')}%")

        st.info(garch.get("interpretation", ""))

        if vol_df is not None and not vol_df.empty:
            vol_df["date"] = pd.to_datetime(vol_df["date"])
            vol_filtered = vol_df[
                (vol_df["date"] >= start_date) & (vol_df["date"] <= end_date)
            ]

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_filtered["date"],
                y=vol_filtered.iloc[:, 1],  # conditional volatility column
                mode="lines",
                name="Conditional Volatility",
                fill="tozeroy",
                fillcolor="rgba(255,127,14,0.3)",
                line=dict(color="#ff7f0e", width=1.5),
            ))
            fig_vol.update_layout(
                yaxis_title="Conditional Volatility (%)",
                xaxis_title="Date", height=300,
            )
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning(f"No GARCH results available for {selected_spice}.")

    # Compare volatility across all spices
    with st.expander("Compare Volatility Across All Spices"):
        all_garch = load_garch_results()
        if all_garch:
            comp_data = []
            for s, g in all_garch.items():
                comp_data.append({
                    "Spice": s,
                    "α (ARCH)": g.get("alpha"),
                    "β (GARCH)": g.get("beta"),
                    "Persistence": g.get("persistence"),
                    "Mean Vol": g.get("mean_cond_vol"),
                    "Max Vol": g.get("max_cond_vol"),
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)


# ─── Panel 3: Cross-Correlation & Granger Causality ─────────────────────────

if show_correlation:
    st.header("Price Volatility ↔ Adulteration Correlation")

    xcorr = load_cross_correlations().get(selected_spice, {})
    granger = load_granger_results().get(selected_spice, {})

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Cross-Correlation (Lag Plot)")
        if xcorr and xcorr.get("lag_correlations"):
            lags_df = pd.DataFrame(xcorr["lag_correlations"])

            fig_xcorr = go.Figure()
            colors = ["#2ca02c" if r["significant"] else "#d62728" for _, r in lags_df.iterrows()]
            fig_xcorr.add_trace(go.Bar(
                x=lags_df["lag"], y=lags_df["correlation"],
                marker_color=colors,
                hovertemplate="Lag: %{x} months<br>r: %{y:.3f}<extra></extra>",
            ))
            fig_xcorr.add_hline(y=0, line_dash="solid", line_color="gray")
            fig_xcorr.update_layout(
                xaxis_title="Lag (months; +ve = volatility leads adulteration)",
                yaxis_title="Pearson Correlation",
                height=350,
            )
            st.plotly_chart(fig_xcorr, use_container_width=True)

            if xcorr.get("best_lag"):
                bl = xcorr["best_lag"]
                st.success(
                    f"**Strongest positive correlation** at lag = **{bl['lag']} months** "
                    f"(r = {bl['correlation']:.3f}, p = {bl['p_value']:.3f})"
                )
        else:
            st.info("Cross-correlation data not available.")

    with col_right:
        st.subheader("Granger Causality Test")
        if granger and granger.get("lag_results"):
            gc_df = pd.DataFrame(granger["lag_results"])
            gc_df["Result"] = gc_df["significant"].map({True: "Significant", False: "Not Sig."})
            st.dataframe(gc_df[["lag", "f_statistic", "p_value", "Result"]], use_container_width=True)

            if granger.get("any_significant"):
                best = granger["best_lag"]
                st.success(
                    f"**Granger causality confirmed** at lag {best['lag']} "
                    f"(F = {best['f_statistic']:.2f}, p = {best['p_value']:.4f}). "
                    f"Price volatility helps predict future adulteration incidence."
                )
            else:
                st.warning(
                    "Granger causality not statistically significant at p < 0.05. "
                    "This may reflect data limitations (adulteration data granularity) "
                    "rather than absence of a real relationship."
                )
        else:
            st.info("Granger causality data not available.")

    # Dual-axis time series plot
    if not df_spice.empty and not df_adult_spice.empty:
        with st.expander("Price vs Adulteration Incidents (Dual-Axis)"):
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(
                go.Scatter(x=df_spice["date"], y=df_spice["modal_price_mean"],
                           name="Price (₹/q)", line=dict(color="#1f77b4")),
                secondary_y=False,
            )
            fig_dual.add_trace(
                go.Bar(x=df_adult_spice["date"], y=df_adult_spice["incident_count"],
                       name="Adulteration Incidents", marker_color="rgba(214,39,40,0.5)"),
                secondary_y=True,
            )
            fig_dual.update_yaxes(title_text="Price (₹/Quintal)", secondary_y=False)
            fig_dual.update_yaxes(title_text="Incident Count", secondary_y=True)
            fig_dual.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_dual, use_container_width=True)

    # Source-level breakdown (stacked bar chart)
    has_sources = all(col in df_adult_spice.columns for col in
                      ["fda_enforcement_count", "rasff_notification_count", "fssai_failure_count"])
    if not df_adult_spice.empty and has_sources:
        with st.expander("Adulteration Data — Source Breakdown"):
            st.caption("Incidents by monitoring agency: US FDA (enforcement), EU RASFF (border notifications), FSSAI (domestic surveillance)")
            fig_src = go.Figure()
            fig_src.add_trace(go.Bar(
                x=df_adult_spice["date"], y=df_adult_spice["fda_enforcement_count"],
                name="US FDA Enforcement", marker_color="#1f77b4",
            ))
            fig_src.add_trace(go.Bar(
                x=df_adult_spice["date"], y=df_adult_spice["rasff_notification_count"],
                name="EU RASFF Notifications", marker_color="#ff7f0e",
            ))
            fig_src.add_trace(go.Bar(
                x=df_adult_spice["date"], y=df_adult_spice["fssai_failure_count"],
                name="FSSAI Domestic Failures", marker_color="#2ca02c",
            ))
            fig_src.update_layout(
                barmode="stack", height=350,
                yaxis_title="Incident Count", xaxis_title="Date",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_src, use_container_width=True)

            # Data quality indicator
            if "fssai_data_quality" in df_adult_spice.columns:
                exact_pct = (df_adult_spice["fssai_data_quality"] == "exact").mean() * 100
                st.info(f"**Data quality:** {exact_pct:.0f}% of FSSAI rows are from exact Parliamentary/official figures. "
                        f"FDA records are real scraped data from openFDA API. "
                        f"RASFF figures are annual totals from published EU Commission reports, distributed monthly.")


# ─── Panel 4: Forecast ──────────────────────────────────────────────────────

if show_forecast:
    st.header("Price Forecast (Actual vs Predicted)")

    fc_results = load_forecast_results().get(selected_spice, {})

    if fc_results:
        best_model = fc_results.get("best_model", "hybrid")
        best_mape = fc_results.get("best_mape", "N/A")

        # Model selector
        available_models = [k for k in ["sarima", "lstm", "hybrid"] if k in fc_results]
        selected_model = st.selectbox(
            "Select Forecast Model",
            available_models,
            index=available_models.index(best_model) if best_model in available_models else 0,
            format_func=lambda x: f"{x.upper()} {'(Best)' if x == best_model else ''}",
        )

        model_data = fc_results[selected_model]

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE", f"{model_data.get('mape', 'N/A')}%")
        col2.metric("RMSE", f"₹{model_data.get('rmse', 'N/A'):,.0f}" if model_data.get('rmse') else "N/A")
        col3.metric("Test Months", model_data.get("test_size", "N/A"))

        # Forecast plot
        if model_data.get("dates") and model_data.get("actual") and model_data.get("predicted"):
            dates = pd.to_datetime(model_data["dates"])
            actual = model_data["actual"]
            predicted = model_data["predicted"]

            fig_fc = go.Figure()

            # Show full historical price before test period
            full_series = df_monthly[df_monthly["spice"] == selected_spice].sort_values("date")
            train_end = dates[0]
            historical = full_series[full_series["date"] < train_end]

            fig_fc.add_trace(go.Scatter(
                x=historical["date"], y=historical["modal_price_mean"],
                mode="lines", name="Historical (Train)",
                line=dict(color="#1f77b4", width=1.5),
            ))

            fig_fc.add_trace(go.Scatter(
                x=dates, y=actual,
                mode="lines+markers", name="Actual (Test)",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=4),
            ))

            fig_fc.add_trace(go.Scatter(
                x=dates, y=predicted,
                mode="lines+markers", name=f"Predicted ({selected_model.upper()})",
                line=dict(color="#d62728", width=2, dash="dash"),
                marker=dict(size=4),
            ))

            # Also show SARIMA baseline if viewing hybrid
            if selected_model == "hybrid" and model_data.get("sarima_predicted"):
                fig_fc.add_trace(go.Scatter(
                    x=dates, y=model_data["sarima_predicted"],
                    mode="lines", name="SARIMA Component",
                    line=dict(color="#9467bd", width=1, dash="dot"),
                ))

            fig_fc.add_vline(x=train_end, line_dash="dash", line_color="gray",
                             annotation_text="Train ← | → Test")
            fig_fc.update_layout(
                yaxis_title="Price (₹/Quintal)", xaxis_title="Date",
                height=400, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        # Model comparison table
        with st.expander("Model Comparison"):
            comp_rows = []
            for m in ["sarima", "lstm", "hybrid"]:
                if m in fc_results:
                    comp_rows.append({
                        "Model": m.upper(),
                        "MAPE (%)": fc_results[m].get("mape"),
                        "RMSE (₹)": fc_results[m].get("rmse"),
                        "Best?": "*" if m == best_model else "",
                    })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    else:
        st.warning(f"No forecast results available for {selected_spice}.")


# ─── Panel 5: Summary & Insights ────────────────────────────────────────────

if show_summary:
    st.header("Summary & Policy Insights")

    garch = load_garch_results().get(selected_spice, {})
    xcorr = load_cross_correlations().get(selected_spice, {})
    granger = load_granger_results().get(selected_spice, {})
    breaks = load_structural_breaks().get(selected_spice, [])
    fc = load_forecast_results().get(selected_spice, {})

    # Auto-generated summary
    summary_parts = [f"### {selected_spice} — Key Findings\n"]

    # Price overview
    if not df_spice.empty:
        avg_price = df_spice["modal_price_mean"].mean()
        price_range = f"₹{df_spice['modal_price_mean'].min():,.0f} – ₹{df_spice['modal_price_mean'].max():,.0f}"
        summary_parts.append(f"**Average price:** ₹{avg_price:,.0f}/quintal (range: {price_range})")

    # Volatility
    if garch:
        persistence = garch.get("persistence", 0)
        vol_level = "HIGH" if persistence > 0.9 else "MODERATE" if persistence > 0.7 else "LOW"
        summary_parts.append(
            f"\n**Volatility:** {vol_level} persistence (α+β = {persistence:.3f}). "
            f"{garch.get('interpretation', '')}"
        )

    # Breaks
    if breaks:
        up_breaks = [b for b in breaks if b["direction"] == "up"]
        summary_parts.append(
            f"\n**Structural breaks:** {len(breaks)} detected "
            f"({len(up_breaks)} upward, {len(breaks) - len(up_breaks)} downward)."
        )

    # Correlation
    if xcorr and xcorr.get("best_lag"):
        bl = xcorr["best_lag"]
        lag_val = bl.get("lag", 0)
        corr_val = bl.get("correlation", 0.0)
        lag_note = f" This suggests that adulteration incidents tend to rise approximately {lag_val} months after price volatility spikes." if lag_val > 0 else ""
        summary_parts.append(
            f"\n**Price-Adulteration Link:** Strongest correlation at **{lag_val}-month lag** "
            f"(r = {corr_val:.3f}).{lag_note}"
        )

    # Granger
    if granger:
        if granger.get("any_significant"):
            summary_parts.append(
                "\n**Granger causality:** Confirmed — price volatility statistically "
                "helps predict future adulteration rates."
            )
        else:
            summary_parts.append(
                "\n**Granger causality:** Not significant at p < 0.05. "
                "Relationship is suggestive but not strong enough for causal claims "
                "given current data granularity."
            )

    # Forecast
    if fc and fc.get("best_model"):
        summary_parts.append(
            f"\n**Best forecast model:** {fc['best_model'].upper()} "
            f"(MAPE = {fc['best_mape']:.1f}%)"
        )

    # Policy framing
    summary_parts.append(
        "\n---\n"
        "### Policy Implication\n"
        f"When {selected_spice} prices spike or show high volatility, "
        f"FSSAI / State Food Safety Officers should intensify surveillance "
        f"and random sampling of {selected_spice.lower()} products in the following months. "
        f"This predictive signal could support **risk-based inspection targeting** — "
        f"allocating limited testing resources to commodities and regions where "
        f"economic pressure for adulteration is highest."
    )

    st.markdown("\n\n".join(summary_parts))

    # Data limitations
    with st.expander("Data Limitations & Caveats"):
        st.markdown("""
        **Important caveats for interpreting these results:**

        1. **Adulteration data is multi-sourced with varying granularity:**
           - **US FDA:** Real enforcement records scraped from openFDA API (recalls, seizures).
             These are *domestic US actions* on spice products — not comprehensive of all Indian spice issues.
           - **EU RASFF:** Annual aggregate notification counts from published EU Commission RASFF
             Annual Reports, distributed monthly using seasonal weights. Individual notification
             records are behind the RASFF Window portal (EU Login required).
           - **FSSAI:** Domestic non-conformance rates from Parliamentary replies (Rajya Sabha Q.1907,
             Lok Sabha Q.1252). Exact data available only for FY 2022-24; earlier years extrapolated.

        2. **Correlation ≠ Causation:** Even with Granger causality, we establish *temporal precedence*
           and *predictive power*, not definitive causal mechanisms. Multiple confounders exist
           (monsoon, policy changes, enforcement drive schedules).

        3. **Export vs. Domestic:** EU RASFF/FDA data reflects export-side rejection rates.
           Domestic adulteration rates in unorganized retail markets are likely higher but unmeasured.

        4. **FSSAI monthly distribution is estimated:** FSSAI reports annual totals; the monthly
           distribution within each year is modeled, not observed. Only the annual failure rates
           for FY 2022-24 are exact.

        5. **FDA data covers all spice products, not just Indian-origin.** We filter by spice
           keywords but cannot always determine country of origin from the enforcement records.
        """)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")

