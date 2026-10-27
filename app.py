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

import llm

from bifs_rnn_lstm_gru_comparison import (MODEL_NAMES, Config, build_datasets, evaluate,
                                          forecast_next_day, load_prices, to_price)

def _find_results() -> str:
    """results/ if present, otherwise the repo root (flat upload)."""
    env = os.environ.get("BIFS_RESULTS")
    for c in ([env] if env else []) + ["results", "."]:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    return env or "results"


RESULTS = _find_results()


def _models_dir() -> str:
    for d in (os.path.join(RESULTS, "models"), "models", RESULTS):
        if os.path.exists(os.path.join(d, "scaler.joblib")):
            return d
    return os.path.join(RESULTS, "models")
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
    mdir = _models_dir()
    models = {n: keras.models.load_model(
        os.path.join(mdir, n.lower().replace(" ", "_") + ".keras")) for n in MODEL_NAMES}
    scaler = joblib.load(os.path.join(mdir, "scaler.joblib"))
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
    st.error("No config.json found (looked in results/ and the repo root). Upload the files "
             "from your Colab results folder to GitHub, then reboot the app.")
    st.stop()

cfg, models, scaler = load_artifacts(RESULTS)
hist_df = cached_prices(json.dumps(cfg.__dict__, default=list))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Forecast", "Backtest", "Model comparison", "Risk (VaR / ES)", "AI Insights"])

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

# ------------------------------------------------------------ AI Insights
SYSTEM_PROMPT = (
    "You are a careful teaching assistant for an MPhil data-science project that compares "
    "Simple RNN, LSTM and GRU networks for one-day-ahead stock price forecasting. "
    "Use ONLY the numbers in the CONTEXT block. If something is not in the context, say you do "
    "not know instead of guessing. Explain clearly in plain language, mention that the naive "
    "baseline (tomorrow = today) is the benchmark, be honest about weak results, and never give "
    "investment advice.")

QUICK = {
    "Explain the results simply": "Explain these results to a non-technical reader in about 150 words.",
    "Draft a report conclusion": "Write a formal conclusion paragraph (about 120 words) for the project report.",
    "Suggest improvements": "Suggest 5 concrete, realistic improvements to the experiment and why each helps.",
    "Viva questions": "Give 6 likely examiner questions about this project, each with a short model answer.",
}


def get_secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


@st.cache_data(ttl=600, show_spinner=False)
def cached_models(provider: str, key: str):
    return llm.list_models(provider, key)


def pick_default(provider: str, models: list) -> int:
    prefs = (["llama-3.3-70b-versatile", "openai/gpt-oss-120b"] if provider == "Groq"
             else ["flash-lite", "flash"])
    for p in prefs:
        for i, m in enumerate(models):
            if (m == p) if provider == "Groq" else (p in m):
                return i
    return 0


def build_context() -> str:
    parts = ["CONFIG: ticker={}, data {} to {}, look_back={}, split train/val/test={}/{}/{}, "
             "target_mode={}, units={}, dropout={}, seeds={}".format(
                 cfg.ticker, cfg.start_date, cfg.end_date, cfg.look_back, cfg.train_frac,
                 cfg.val_frac, round(1 - cfg.train_frac - cfg.val_frac, 2), cfg.target_mode,
                 cfg.units, cfg.dropout, list(cfg.seeds))]
    p = os.path.join(RESULTS, "metrics_summary.csv")
    if os.path.exists(p):
        parts.append("TEST METRICS in USD (mean/std over seeds; Skill_vs_naive>0 means better than "
                     "naive):\n" + pd.read_csv(p, index_col=0).round(4).to_csv())
    r = np.diff(np.log(hist_df["Close"].to_numpy(float)))
    var99 = np.quantile(-r, 0.99)
    parts.append(f"RISK: historical 1-day 99% VaR = {var99:.2%}, ES = {(-r)[(-r) >= var99].mean():.2%} "
                 "of position (daily log-returns).")
    return "\n\n".join(parts)


with tab5:
    st.subheader("AI Insights")
    st.caption("An AI assistant (Groq or Google Gemini) that explains the project results. "
               "Only the metrics summary and your question are sent to the provider, never your key to this app's repo.")
    provider = st.radio("Provider", ["Groq", "Gemini"], horizontal=True)
    secret_name = "GROQ_API_KEY" if provider == "Groq" else "GEMINI_API_KEY"
    key = get_secret(secret_name)
    if key:
        st.success(f"Using {secret_name} from app secrets.")
    else:
        key = st.text_input(f"{provider} API key", type="password",
                            help="Not stored. For a permanent setup add it to the app's Secrets.")
    if not key:
        st.info("Add your API key to enable this tab. Groq keys: console.groq.com/keys. "
                "Gemini keys: aistudio.google.com/apikey.")
    else:
        try:
            models = cached_models(provider, key)
        except llm.LLMError as e:
            models = []
            st.error(str(e))
        if models:
            model = st.selectbox("Model", models, index=pick_default(provider, models))
        else:
            model = st.text_input("Model name (type it manually)")

        if "ai_chat" not in st.session_state:
            st.session_state.ai_chat = []
        cols = st.columns(len(QUICK))
        asked = None
        for c, (label, prompt) in zip(cols, QUICK.items()):
            if c.button(label, width="stretch"):
                asked = prompt
        typed = st.chat_input("Ask a question about the results")
        asked = typed or asked
        if asked:
            st.session_state.ai_chat.append({"role": "user", "content": asked})
            msgs = [{"role": "user", "content": "CONTEXT:\n" + build_context()}] + \
                   st.session_state.ai_chat[-8:]
            try:
                with st.spinner("Thinking..."):
                    reply = llm.chat(provider, key, model, SYSTEM_PROMPT, msgs)
                st.session_state.ai_chat.append({"role": "assistant", "content": reply})
            except llm.LLMError as e:
                st.session_state.ai_chat.pop()
                st.error(str(e))
        for m in st.session_state.ai_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if st.session_state.ai_chat and st.button("Clear conversation"):
            st.session_state.ai_chat = []
            st.rerun()
        st.caption("AI answers can be wrong; check them against the tables. Not financial advice.")
