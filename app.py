"""
BIFS dashboard (Streamlit).
Run the experiment first (it creates results/), then:  streamlit run app.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow import keras

from bifs_rnn_lstm_gru_comparison import (MODEL_NAMES, Config, build_datasets, evaluate,
                                          forecast_next_day, load_prices, to_price)

RESULTS = os.environ.get("BIFS_RESULTS", "results")
COLORS = {"Simple RNN": "#ff7f0e", "LSTM": "#d62728", "GRU": "#2ca02c",
          "Naive": "#7f7f7f", "Actual": "#000000"}

st.set_page_config(page_title="BIFS Forecasting", layout="wide")
st.title("BIFS: Business Insight Forecasting System")
st.caption("RNN vs LSTM vs GRU one-day-ahead price forecasts, with a naive baseline and risk measures. "
           "Educational project; not financial advice.")


@st.cache_resource
def load_artifacts(results: str):
    with open(os.path.join(results, "config.json")) as f:
        d = json.load(f)
    d["seeds"] = tuple(d["seeds"])
    cfg = Config(**d)
    cfg.show_plots = False
    models = {n: keras.models.load_model(
        os.path.join(results, "models", n.lower().replace(" ", "_") + ".keras")) for n in MODEL_NAMES}
    scaler = joblib.load(os.path.join(results, "models", "scaler.joblib"))
    return cfg, models, scaler


@st.cache_data
def cached_prices(cfg_json: str) -> pd.DataFrame:
    d = json.loads(cfg_json)
    d["seeds"] = tuple(d["seeds"])
    return load_prices(Config(**d))


@st.cache_data(ttl=3600)
def latest_prices(ticker: str):
    """Most recent ~6 months from Yahoo Finance; None if unavailable."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Close"]].dropna().astype(float)
    except Exception:
        return None


if not os.path.exists(os.path.join(RESULTS, "config.json")):
    st.error(f"No trained results found in '{RESULTS}/'. Run the notebook or "
             "`python bifs_rnn_lstm_gru_comparison.py` first.")
    st.stop()

cfg, models, scaler = load_artifacts(RESULTS)
hist_df = cached_prices(json.dumps(cfg.__dict__, default=list))

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Backtest", "Model comparison", "Risk (VaR / ES)"])

# ---------------------------------------------------------------- Forecast
with tab1:
    live = latest_prices(cfg.ticker)
    data = live if live is not None and len(live) > cfg.look_back + 1 else hist_df
    st.write(f"**{cfg.ticker}** - data source: "
             f"{'Yahoo Finance (live)' if data is live else 'cached training data'}, "
             f"last date {data.index[-1].date()}")
    prices = data["Close"].to_numpy(float)
    last = prices[-1]
    if cfg.target_mode == "price" and last > scaler.data_max_[0]:
        st.warning("The latest price is above the highest price seen in training. "
                   "Scaled-price models extrapolate poorly here; consider retraining or "
                   "using target_mode='return'.")
    fc = {m: forecast_next_day(models[m], scaler, prices, cfg) for m in MODEL_NAMES}
    cols = st.columns(4)
    cols[0].metric("Last close", f"{last:,.2f}")
    for c, m in zip(cols[1:], MODEL_NAMES):
        c.metric(f"{m} next day", f"{fc[m]:,.2f}", f"{fc[m] - last:+.2f}")
    nxt = data.index[-1] + pd.offsets.BDay(1)
    shown = data.iloc[-90:]
    fig = go.Figure(go.Scatter(x=shown.index, y=shown["Close"], name="Close",
                               line=dict(color="black")))
    for m in MODEL_NAMES:
        fig.add_trace(go.Scatter(x=[nxt], y=[fc[m]], mode="markers", name=m,
                                 marker=dict(size=12, color=COLORS[m])))
    fig.update_layout(height=420, yaxis_title="Price (USD)", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.caption("Models are trained on the cached period; one-step-ahead forecasts only.")

# ---------------------------------------------------------------- Backtest
with tab2:
    prices_h = hist_df["Close"].to_numpy(float)
    _, _, test, _ = build_datasets(prices_h, hist_df.index.to_numpy(), cfg)
    chosen = st.multiselect("Models", MODEL_NAMES, default=MODEL_NAMES)
    zoom = st.slider("Show last N test days", 30, len(test.y), len(test.y))
    fig = go.Figure()
    sl = slice(-zoom, None)
    fig.add_trace(go.Scatter(x=test.dates[sl], y=test.true_price[sl], name="Actual",
                             line=dict(color=COLORS["Actual"], width=3)))
    fig.add_trace(go.Scatter(x=test.dates[sl], y=test.prev_price[sl], name="Naive",
                             line=dict(color=COLORS["Naive"], dash="dot")))
    rows = {"Naive": evaluate(test.true_price, test.prev_price, test.prev_price)}
    for m in chosen:
        pred = to_price(models[m].predict(test.X, verbose=0), test, scaler, cfg.target_mode)
        rows[m] = evaluate(test.true_price, pred, test.prev_price)
        fig.add_trace(go.Scatter(x=test.dates[sl], y=pred[sl], name=m,
                                 line=dict(color=COLORS[m], dash="dash")))
    fig.update_layout(height=450, yaxis_title="Price (USD)", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows).T.round(3), width="stretch")
    st.caption("Metrics are for the saved (best-validation) seed on the full test period, in USD.")

# ------------------------------------------------------- Model comparison
with tab3:
    path = os.path.join(RESULTS, "metrics_summary.csv")
    if os.path.exists(path):
        summ = pd.read_csv(path, index_col=0)
        st.subheader("Test metrics, mean and std over seeds")
        st.dataframe(summ.round(4), width="stretch")
        order = ["Naive"] + MODEL_NAMES
        fig = go.Figure(go.Bar(x=order, y=summ.loc[order, "RMSE_mean"],
                               error_y=dict(type="data", array=summ.loc[order, "RMSE_std"].fillna(0)),
                               marker_color=[COLORS[o] for o in order]))
        fig.update_layout(yaxis_title="Test RMSE (USD)", height=380)
        st.plotly_chart(fig, width="stretch")
        st.write("**Skill vs naive** (positive = better than 'tomorrow = today'):")
        st.write(summ["Skill_vs_naive"].map("{:+.1%}".format).to_frame().T)
    for img in ("training_history.png", "residuals_analysis.png"):
        p = os.path.join(RESULTS, img)
        if os.path.exists(p):
            st.image(p)

# -------------------------------------------------------------- Risk
with tab4:
    st.write("Historical-simulation risk of **daily log-returns** of the cached price series.")
    conf = st.slider("Confidence level", 0.90, 0.995, 0.99, 0.005)
    horizon = st.number_input("Horizon (days, square-root-of-time approximation)", 1, 30, 1)
    rets = np.diff(np.log(hist_df["Close"].to_numpy(float)))
    losses = -rets
    var = np.quantile(losses, conf)
    es = losses[losses >= var].mean()
    k = np.sqrt(horizon)
    c1, c2 = st.columns(2)
    c1.metric(f"VaR {conf:.1%} ({horizon}d)", f"{var * k:.2%} of position")
    c2.metric(f"Expected Shortfall {conf:.1%} ({horizon}d)", f"{es * k:.2%} of position")
    fig = go.Figure(go.Histogram(x=losses * 100, nbinsx=60, marker_color="#4c72b0"))
    fig.add_vline(x=var * 100, line_color="red", annotation_text="VaR")
    fig.add_vline(x=es * 100, line_color="black", annotation_text="ES")
    fig.update_layout(xaxis_title="Daily loss (%)", yaxis_title="Days", height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("VaR = loss quantile; ES = average loss beyond VaR. Basel's FRTB uses ES at 97.5 %. "
               "Square-root-of-time scaling is a rough approximation.")
