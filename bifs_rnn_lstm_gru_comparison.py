# %% [markdown]
# # BIFS: RNN vs LSTM vs GRU for Stock Price Forecasting
# **MPhil project, corrected version.**
#
# One-day-ahead forecasting of the closing price with three recurrent networks,
# evaluated honestly:
# * scaler fitted on training data only (no leakage)
# * chronological train / validation / test split
# * identical architectures for all three models
# * metrics computed in real USD
# * naive "tomorrow = today" baseline
# * several random seeds, reported as mean +/- std

# %%
import gc
import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import NamedTuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore", category=FutureWarning)


# %% [markdown]
# ## 1. Configuration
# Everything you may want to change is here.
# * `target_mode="price"`: the model predicts the (MinMax-scaled) price, as in the
#   original project.
# * `target_mode="return"`: the model predicts the next-day log-return and the price is
#   rebuilt as `P_t = P_(t-1) * exp(r_t)`. This avoids the "test prices are outside the
#   training range" problem of an upward-trending stock.

# %%
@dataclass
class Config:
    ticker: str = "AAPL"
    start_date: str = "2021-01-01"
    end_date: str = "2026-01-01"
    look_back: int = 15            # use previous 15 days to predict the next day
    train_frac: float = 0.70
    val_frac: float = 0.15         # test fraction = 1 - train - val
    target_mode: str = "price"     # "price" or "return"
    units: int = 50
    dropout: float = 0.2
    batch_size: int = 16
    epochs: int = 100              # upper limit; early stopping decides
    patience: int = 15
    seeds: tuple = (42, 43, 44, 45, 46)
    data_dir: str = "data"
    output_dir: str = "results"
    force_download: bool = False
    show_plots: bool = True


MODEL_NAMES = ["Simple RNN", "LSTM", "GRU"]
CELLS = {"Simple RNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}


# %% [markdown]
# ## 2. Data loading
# Prices are downloaded once and cached as a CSV (`data/`), so every re-run uses
# exactly the same data. `auto_adjust=True` means the Close price is adjusted for
# splits and dividends.

# %%
def load_prices(cfg: Config) -> pd.DataFrame:
    """Return a DataFrame with one column, 'Close', indexed by date."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    fname = f"{cfg.ticker}_{cfg.start_date}_{cfg.end_date}.csv"
    cache = os.path.join(cfg.data_dir, fname)
    # look in data/, the repo root and results/ (so a flat GitHub upload also works)
    found = next((p for p in (cache, fname, os.path.join("results", fname)) if os.path.exists(p)), None)

    if found and not cfg.force_download:
        df = pd.read_csv(found, index_col=0, parse_dates=True)
        print(f"Loaded cached data: {found}")
    else:
        import yfinance as yf

        print(f"Downloading {cfg.ticker} {cfg.start_date} -> {cfg.end_date} ...")
        raw = yf.download(cfg.ticker, start=cfg.start_date, end=cfg.end_date,
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError(
                "No data returned by Yahoo Finance. Check the ticker, the dates and "
                "your internet connection, or upload the cached CSV "
                f"'{fname}' (Date,Close columns) to the repo's data/ folder.")
        if isinstance(raw.columns, pd.MultiIndex):      # newer yfinance versions
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close"]].copy()
        df.to_csv(cache)
        print(f"Saved to cache: {cache}")

    df = df[["Close"]].astype(float).dropna().sort_index()
    if (df["Close"] <= 0).any():
        raise ValueError("Non-positive prices found; cannot continue.")
    if len(df) < 200:
        raise ValueError(f"Only {len(df)} rows. Need more data.")
    print(f"{len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
    return df


# %% [markdown]
# ## 3. Leak-free preprocessing
# **Scaling** (price mode): x' = (x - x_min) / (x_max - x_min), where x_min and x_max come
# from the *training* period only.
#
# **Windows:** a sample for target day *i* uses days *i-15 ... i-1* as input and day *i*
# as the answer. The split is done by the **target day**, so every validation / test
# target is predicted with a full 15-day window (no days are thrown away at the split
# edges) and no input window ever contains the target or any future day.

# %%
class Split(NamedTuple):
    X: np.ndarray            # (samples, look_back, 1)
    y: np.ndarray            # scaled target
    prev_price: np.ndarray   # real price on the day before the target
    true_price: np.ndarray   # real price on the target day (USD)
    dates: np.ndarray


def build_datasets(prices: np.ndarray, dates: np.ndarray, cfg: Config):
    n, lb = len(prices), cfg.look_back
    train_end = int(n * cfg.train_frac)
    val_end = train_end + int(n * cfg.val_frac)
    first = lb + 1                      # first usable target index (both modes)

    if cfg.target_mode == "price":
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(prices[:train_end].reshape(-1, 1))
        series = scaler.transform(prices.reshape(-1, 1)).ravel()
    elif cfg.target_mode == "return":
        rets = np.concatenate([[0.0], np.diff(np.log(prices))])   # rets[i] = ln(P_i / P_(i-1))
        scaler = StandardScaler().fit(rets[1:train_end].reshape(-1, 1))
        series = scaler.transform(rets.reshape(-1, 1)).ravel()
    else:
        raise ValueError("target_mode must be 'price' or 'return'")

    def make(lo: int, hi: int) -> Split:
        idx = np.arange(max(lo, first), hi)
        X = np.stack([series[i - lb:i] for i in idx])[..., None].astype("float32")
        return Split(X, series[idx].astype("float32"), prices[idx - 1], prices[idx], dates[idx])

    train, val, test = make(0, train_end), make(train_end, val_end), make(val_end, n)
    for name, s in (("train", train), ("val", val), ("test", test)):
        if len(s.y) == 0:
            raise ValueError(f"{name} split is empty; adjust the fractions.")
    print(f"Samples  train={len(train.y)}  val={len(val.y)}  test={len(test.y)}")
    return train, val, test, scaler


def to_price(pred_scaled, split: Split, scaler, mode: str) -> np.ndarray:
    """Convert model output back to real USD prices."""
    inv = scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    return inv if mode == "price" else split.prev_price * np.exp(inv)


# %% [markdown]
# ## 4. Models
# All three use the **same** structure: 2 recurrent layers (50 units) + dropout +
# Dense(25, ReLU) + Dense(1). Only the cell type differs, so the comparison is fair.

# %%
def build_model(name: str, cfg: Config) -> keras.Model:
    cell = CELLS[name]
    model = keras.Sequential([
        layers.Input(shape=(cfg.look_back, 1)),
        cell(cfg.units, return_sequences=True),
        layers.Dropout(cfg.dropout),
        cell(cfg.units),
        layers.Dropout(cfg.dropout),
        layers.Dense(25, activation="relu"),
        layers.Dense(1),
    ], name=name.replace(" ", "_"))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train_model(model, train: Split, val: Split, cfg: Config):
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                         restore_best_weights=True)
    return model.fit(train.X, train.y, validation_data=(val.X, val.y),
                     batch_size=cfg.batch_size, epochs=cfg.epochs,
                     callbacks=[stop], verbose=0)


# %% [markdown]
# ## 5. Evaluation (in real USD)
# * RMSE = sqrt( mean( (y - y_hat)^2 ) )
# * MAE = mean( |y - y_hat| )
# * MAPE = mean( |y - y_hat| / y ) x 100
# * R2 = 1 - SS_res / SS_tot
# * DirAcc = % of days where the predicted direction (up/down vs. yesterday) is correct
# * Skill vs naive = 1 - RMSE_model / RMSE_naive (positive means better than "tomorrow = today")

# %%
def evaluate(y_true, y_pred, prev) -> dict:
    is_naive = np.allclose(y_pred, prev)
    direction = (np.sign(y_pred - prev) == np.sign(y_true - prev)).mean() * 100
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
        "DirAcc": float("nan") if is_naive else float(direction),
    }


# %% [markdown]
# ## 6. Plots

# %%
COLORS = {"Simple RNN": "tab:orange", "LSTM": "tab:red", "GRU": "tab:green"}


def _finish(fig, path: str, cfg: Config):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    print(f"saved {path}")


def plot_history(best: dict, cfg: Config):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, MODEL_NAMES):
        h = best[name]["hist"]
        ax.plot(h["loss"], label="Train loss", lw=2)
        ax.plot(h["val_loss"], label="Validation loss", lw=2)
        ax.set_title(f"{name} (seed {best[name]['seed']})", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled units)")
        ax.legend(); ax.grid(alpha=0.3)
    _finish(fig, os.path.join(cfg.output_dir, "training_history.png"), cfg)


def plot_predictions(test: Split, best: dict, cfg: Config):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, last in zip(axes, (None, 60)):
        sl = slice(-last, None) if last else slice(None)
        ax.plot(test.dates[sl], test.true_price[sl], "k", lw=2.5, label="Actual")
        ax.plot(test.dates[sl], test.prev_price[sl], color="gray", lw=1.2, ls=":",
                label="Naive (yesterday's price)")
        for name in MODEL_NAMES:
            ax.plot(test.dates[sl], best[name]["pred"][sl], color=COLORS[name], lw=1.8,
                    ls="--", alpha=0.85, label=name)
        ax.set_ylabel("Price (USD)"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("Test period: full" if last is None else f"Test period: last {last} days (zoom)",
                     fontweight="bold")
    _finish(fig, os.path.join(cfg.output_dir, "predictions_comparison.png"), cfg)


def plot_residuals(test: Split, best: dict, cfg: Config):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, MODEL_NAMES):
        res = test.true_price - best[name]["pred"]
        ax.hist(res, bins=30, color=COLORS[name], edgecolor="black", alpha=0.75)
        ax.axvline(0, color="black", ls="--", lw=2)
        ax.set_title(f"{name}: residuals (mean {res.mean():.2f} USD)", fontweight="bold")
        ax.set_xlabel("Actual - Predicted (USD)"); ax.set_ylabel("Frequency"); ax.grid(alpha=0.3)
    _finish(fig, os.path.join(cfg.output_dir, "residuals_analysis.png"), cfg)


def plot_metric_bars(summary: pd.DataFrame, cfg: Config):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["Naive"] + MODEL_NAMES
    s = summary.loc[order]
    cols = ["gray"] + [COLORS[m] for m in MODEL_NAMES]
    ax.bar(order, s["RMSE_mean"], yerr=s["RMSE_std"].fillna(0), color=cols,
           edgecolor="black", capsize=6)
    ax.set_ylabel("Test RMSE (USD), lower is better")
    ax.set_title("RMSE: mean +/- std over seeds", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, os.path.join(cfg.output_dir, "rmse_comparison.png"), cfg)


# %% [markdown]
# ## 7. Forecast helper (for an app / dashboard)
# Uses a trained model and its scaler to predict the **next trading day** from the most
# recent prices. For learning purposes only. It is not financial advice.

# %%
def forecast_next_day(model, scaler, prices: np.ndarray, cfg: Config) -> float:
    lb = cfg.look_back
    prices = np.asarray(prices, dtype=float)
    if cfg.target_mode == "price":
        window = scaler.transform(prices[-lb:].reshape(-1, 1))
        out = scaler.inverse_transform(model.predict(window[None].astype("float32"), verbose=0))
        return float(out.item())
    rets = np.diff(np.log(prices[-(lb + 1):]))
    window = scaler.transform(rets.reshape(-1, 1))
    r = scaler.inverse_transform(model.predict(window[None].astype("float32"), verbose=0))
    return float(prices[-1] * np.exp(r.item()))


# %% [markdown]
# ## 8. Run the full experiment

# %%
def run_experiment(cfg: Config = None) -> dict:
    cfg = cfg or Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "models"), exist_ok=True)

    df = load_prices(cfg)
    prices = df["Close"].to_numpy(dtype=float)
    dates = df.index.to_numpy()
    train, val, test, scaler = build_datasets(prices, dates, cfg)

    # Naive baseline: tomorrow's price = today's price
    naive = evaluate(test.true_price, test.prev_price, test.prev_price)

    rows, best = [], {}
    for seed in cfg.seeds:
        for name in MODEL_NAMES:
            keras.utils.set_random_seed(seed)
            model = build_model(name, cfg)
            if seed == cfg.seeds[0]:
                print(f"{name}: {model.count_params():,} trainable parameters")
            hist = train_model(model, train, val, cfg)
            val_loss = float(min(hist.history["val_loss"]))
            pred = to_price(model.predict(test.X, verbose=0), test, scaler, cfg.target_mode)
            m = evaluate(test.true_price, pred, test.prev_price)
            rows.append({"Model": name, "Seed": seed, "ValLoss": val_loss,
                         "Epochs": len(hist.history["loss"]), **m})
            print(f"  seed {seed:>3} | {name:<10} | epochs {len(hist.history['loss']):>3} | "
                  f"RMSE {m['RMSE']:.3f} USD | R2 {m['R2']:.3f}")
            # keep the seed with the best VALIDATION loss (never selected on test data)
            if name not in best or val_loss < best[name]["val_loss"]:
                best[name] = {"model": model, "hist": hist.history, "pred": pred,
                              "seed": seed, "val_loss": val_loss}
            else:
                del model
            gc.collect()

    runs = pd.DataFrame(rows)
    metric_cols = ["RMSE", "MAE", "MAPE", "R2", "DirAcc"]
    agg = runs.groupby("Model")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    naive_row = {}
    for c in metric_cols:
        naive_row[f"{c}_mean"], naive_row[f"{c}_std"] = naive[c], 0.0
    summary = pd.concat([agg, pd.DataFrame(naive_row, index=["Naive"])])
    summary["Skill_vs_naive"] = 1 - summary["RMSE_mean"] / naive["RMSE"]

    # ---- print results
    print("\n" + "=" * 72)
    print(f"TEST RESULTS in USD (mean +/- std over {len(cfg.seeds)} seeds)")
    print("=" * 72)
    show = pd.DataFrame(index=summary.index)
    for c, f in (("RMSE", "{:.3f}"), ("MAE", "{:.3f}"), ("MAPE", "{:.2f}"),
                 ("R2", "{:.4f}"), ("DirAcc", "{:.1f}")):
        show[c] = [(f + " +/- " + f).format(m, s) if not np.isnan(m) else "n/a"
                   for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_std"].fillna(0))]
    show["Skill_vs_naive"] = summary["Skill_vs_naive"].map("{:+.1%}".format)
    print(show.to_string())

    ranked = summary["RMSE_mean"].sort_values()
    winner, runner = ranked.index[0], ranked.index[1]
    gap = ranked.iloc[1] - ranked.iloc[0]
    spread = max(summary.loc[winner, "RMSE_std"], summary.loc[runner, "RMSE_std"])
    print(f"\nLowest mean RMSE: {winner}.")
    if winner == "Naive":
        print("The naive baseline beat every neural model. Report this honestly: on this "
              "data the networks add no value over 'tomorrow = today'.")
    elif gap < spread:
        print(f"Gap to {runner} ({gap:.3f} USD) is smaller than the seed-to-seed "
              "variation, so the ranking is not conclusive.")
    print("=" * 72)

    # ---- save everything
    runs.to_csv(os.path.join(cfg.output_dir, "metrics_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(cfg.output_dir, "metrics_summary.csv"))
    for name in MODEL_NAMES:
        best[name]["model"].save(os.path.join(cfg.output_dir, "models",
                                              name.lower().replace(" ", "_") + ".keras"))
    joblib.dump(scaler, os.path.join(cfg.output_dir, "models", "scaler.joblib"))
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    plot_history(best, cfg)
    plot_predictions(test, best, cfg)
    plot_residuals(test, best, cfg)
    plot_metric_bars(summary, cfg)

    print("\nIllustrative next-day forecasts (not financial advice):")
    for name in MODEL_NAMES:
        p = forecast_next_day(best[name]["model"], scaler, prices, cfg)
        print(f"  {name:<10}: {p:8.2f} USD   (last close {prices[-1]:.2f})")

    return {"summary": summary, "runs": runs, "best": best, "test": test,
            "scaler": scaler, "config": cfg}


# %%
if __name__ == "__main__":
    run_experiment(Config())
