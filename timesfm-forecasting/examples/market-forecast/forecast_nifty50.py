#!/usr/bin/env python3
"""
TimesFM Market Data Forecasting — Nifty 50 Example

Demonstrates zero-shot forecasting of the Nifty 50 index using TimesFM.
Fetches real-time data via yfinance (15–20 min delayed, free) and falls back
to a realistic synthetic series when the network is unavailable.

Usage
-----
    # Install extra dependency once:
    pip install yfinance

    # Then run:
    python forecast_nifty50.py

    # Force synthetic data (no network):
    python forecast_nifty50.py --synthetic

    # Different ticker (e.g. S&P 500, Bank Nifty):
    python forecast_nifty50.py --ticker "^GSPC"
    python forecast_nifty50.py --ticker "^NSEBANK"

Outputs (saved to output/)
--------------------------
    nifty50_forecast.png    — 2-panel chart: price history + prediction intervals
    nifty50_forecast.json   — structured forecast + accuracy metrics
    nifty50_forecast.csv    — tabular forecast data

Accuracy expectations
---------------------
TimesFM is a general-purpose foundation model, not purpose-built for finance.
Stock / index forecasting is notoriously hard (efficient market hypothesis).
Treat these forecasts as **statistical extrapolations of recent trend + seasonality**,
NOT as trading signals. See README.md for a detailed capability assessment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"

# Quantile index constants (TimesFM 1.0 / 2.x shared convention)
# index 0 = mean, 1 = q10, 2 = q20 ... 9 = q90
IDX_Q10, IDX_Q20, IDX_Q80, IDX_Q90 = 1, 2, 8, 9

FORECAST_HORIZON = 30  # trading days (~6 calendar weeks)
CONTEXT_DAYS = 252     # ~1 trading year


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _make_synthetic_nifty(n: int = CONTEXT_DAYS, seed: int = 42) -> pd.DataFrame:
    """Return a realistic synthetic Nifty 50 daily-close series.

    Simulates a geometric Brownian motion with:
      - mu  = 0.12  (12 % annual drift, roughly in-line with historical Nifty average)
      - sigma = 0.15 (15 % annual volatility)
    Starting value ≈ 22 000 (approximate Nifty 50 level as of early 2025).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu = 0.12
    sigma = 0.15
    s0 = 22_000.0

    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    prices = s0 * np.exp(np.cumsum(log_returns))
    prices = np.concatenate([[s0], prices])[: n]

    # Build a trading-day date index ending "today"
    today = pd.Timestamp.today().normalize()
    bdays = pd.bdate_range(end=today, periods=n)
    return pd.DataFrame({"close": prices.astype(np.float32)}, index=bdays)


def fetch_nifty_data(ticker: str = "^NSEI", context_days: int = CONTEXT_DAYS) -> pd.DataFrame:
    """Download the last *context_days* trading days from Yahoo Finance.

    Returns a DataFrame with columns: ['close']
    Raises RuntimeError when yfinance is not installed or the download fails.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed.  Run:  pip install yfinance\n"
            "Or use --synthetic to skip network access."
        ) from exc

    # Fetch a bit more than we need to account for holidays
    period_days = int(context_days * 1.5)
    start = pd.Timestamp.today() - pd.Timedelta(days=period_days)

    print(f"  Fetching {ticker} from Yahoo Finance …")
    df_raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if df_raw.empty:
        raise RuntimeError(f"yfinance returned no data for ticker '{ticker}'.")

    # yfinance may return a MultiIndex for single-ticker downloads in newer versions
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.droplevel(1)

    df = df_raw[["Close"]].rename(columns={"Close": "close"}).dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().tail(context_days)
    print(f"  Downloaded {len(df)} trading days  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df.astype(np.float32)


def load_market_data(ticker: str, context_days: int, synthetic: bool) -> tuple[pd.DataFrame, bool]:
    """Return (df, is_synthetic).  Falls back to synthetic on any error."""
    if synthetic:
        print("  [synthetic mode] Generating realistic GBM price series …")
        return _make_synthetic_nifty(context_days), True

    try:
        return fetch_nifty_data(ticker, context_days), False
    except RuntimeError as exc:
        print(f"\n  ⚠️  Real data unavailable: {exc}")
        print("  ⚠️  Falling back to synthetic data.\n")
        return _make_synthetic_nifty(context_days), True


# ---------------------------------------------------------------------------
# TimesFM forecasting
# ---------------------------------------------------------------------------


def run_forecast(
    close_prices: np.ndarray,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Load TimesFM 1.0 and produce (point_forecast, quantile_forecast).

    Returns
    -------
    point_fc  : shape (horizon,)
    quant_fc  : shape (10, horizon)  — indices follow IDX_Q10 etc.
    """
    import timesfm  # noqa: PLC0415

    print("\n  Loading TimesFM 1.0 (200 M parameters, PyTorch) …")
    hparams = timesfm.TimesFmHparams(horizon_len=horizon)
    checkpoint = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
    model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    print(f"  Running zero-shot forecast  (context={len(close_prices)}, horizon={horizon}) …")
    point_out, quant_out = model.forecast(
        [close_prices.astype(np.float32)],
        freq=[0],  # freq=0 → default / daily  (required for TimesFM 1.0)
    )

    point_fc = point_out[0]           # (horizon,)
    quant_fc = quant_out[0].T         # (10, horizon)
    return point_fc, quant_fc


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE, and directional accuracy."""
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # Avoid division by zero for MAPE
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)

    # Directional accuracy: did forecast move in the right direction?
    actual_dir = np.sign(np.diff(np.concatenate([[actual[0] - 1], actual])))
    pred_dir   = np.sign(np.diff(np.concatenate([[actual[0] - 1], predicted])))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "mae":  round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "directional_accuracy_pct": round(dir_acc, 1),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_forecast(
    df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    point_fc: np.ndarray,
    quant_fc: np.ndarray,
    ticker: str,
    is_synthetic: bool,
    accuracy: dict | None,
) -> None:
    """Two-panel chart: candlestick-style history + forecast with PI bands."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    context_values = df["close"].values
    display_context = min(90, len(context_values))  # last 90 days in top panel

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"hspace": 0.45, "height_ratios": [2, 1]},
    )

    source_tag = "(SYNTHETIC)" if is_synthetic else "(Yahoo Finance)"
    fig.suptitle(
        f"TimesFM Zero-Shot Forecast — {ticker} {source_tag}\n"
        f"Context: {len(context_values)} trading days  |  "
        f"Horizon: {FORECAST_HORIZON} trading days (~6 weeks)",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------------
    # Panel 1: historical close + forecast + prediction intervals
    # -----------------------------------------------------------------------
    ctx_x = df.index[-display_context:]
    ctx_y = context_values[-display_context:]

    q10 = quant_fc[IDX_Q10]
    q20 = quant_fc[IDX_Q20]
    q80 = quant_fc[IDX_Q80]
    q90 = quant_fc[IDX_Q90]

    ax1.plot(ctx_x, ctx_y, color="#1a56db", lw=1.8, label=f"{ticker} close (last {display_context}d)")
    ax1.plot(forecast_dates, point_fc, color="#e03030", lw=2.2, marker="o", ms=3.5,
             label="TimesFM point forecast")
    ax1.fill_between(forecast_dates, q10, q90, alpha=0.18, color="#e03030", label="80 % PI (q10–q90)")
    ax1.fill_between(forecast_dates, q20, q80, alpha=0.28, color="#e03030", label="60 % PI (q20–q80)")

    # Divider at last known date
    divider = df.index[-1]
    ax1.axvline(divider, color="#555555", lw=1.5, ls=":")
    ylim_top = ax1.get_ylim()[1] or 1.0
    ax1.text(
        divider, ylim_top,
        "  ← Observed | Forecast →",
        fontsize=8.5, color="#555555", style="italic", va="top",
    )

    # Accuracy annotation (if available)
    if accuracy:
        acc_text = (
            f"Accuracy (hold-out test):\n"
            f"  MAE:  {accuracy['mae']:,.0f}  |  MAPE:    {accuracy['mape_pct']:.1f} %\n"
            f"  RMSE: {accuracy['rmse']:,.0f}  |  Dir Acc: {accuracy['directional_accuracy_pct']:.0f} %"
        )
        ax1.annotate(
            acc_text,
            xy=(0.01, 0.05), xycoords="axes fraction",
            fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="#aaaaaa", alpha=0.9),
        )

    ax1.set_ylabel("Index Level", fontsize=10)
    ax1.legend(ncol=2, fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.22)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # -----------------------------------------------------------------------
    # Panel 2: forecast uncertainty band width over horizon
    # -----------------------------------------------------------------------
    horizon_x = np.arange(1, FORECAST_HORIZON + 1)
    band_80 = q90 - q10
    band_60 = q80 - q20

    ax2.fill_between(horizon_x, 0, band_80, alpha=0.30, color="#e03030", label="80 % PI width")
    ax2.fill_between(horizon_x, 0, band_60, alpha=0.45, color="#e03030", label="60 % PI width")
    ax2.plot(horizon_x, band_80, color="#e03030", lw=1.5)
    ax2.axhline(0, color="black", lw=0.7, alpha=0.5)

    # Annotate relative uncertainty at final horizon step
    last_close = context_values[-1]
    pct_80 = band_80[-1] / last_close * 100
    ax2.annotate(
        f"Day {FORECAST_HORIZON}: ±{pct_80/2:.1f} % range (80 % PI)",
        xy=(FORECAST_HORIZON, band_80[-1]),
        xytext=(FORECAST_HORIZON - 8, band_80[-1] * 1.1),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#555555"),
        color="#555555",
    )

    ax2.set_xlabel("Forecast horizon (trading days)", fontsize=10)
    ax2.set_ylabel("PI width (index points)", fontsize=10)
    ax2.set_title("Forecast Uncertainty Growth", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.22)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "nifty50_forecast.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅  Saved chart: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TimesFM zero-shot forecast of a market index (default: Nifty 50)"
    )
    parser.add_argument("--ticker", default="^NSEI",
                        help="Yahoo Finance ticker symbol (default: ^NSEI = Nifty 50)")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON,
                        help="Forecast horizon in trading days (default: 30)")
    parser.add_argument("--context", type=int, default=CONTEXT_DAYS,
                        help="Historical context length in trading days (default: 252)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip network; use synthetic GBM price data")
    parser.add_argument("--holdout", type=int, default=0,
                        help="Hold out last N days from context for accuracy evaluation "
                             "(0 = no evaluation)")
    args = parser.parse_args(argv)

    print("=" * 66)
    print("  TIMESFM MARKET FORECAST — ZERO-SHOT INDEX FORECASTING")
    print("=" * 66)
    print(f"\n  Ticker:  {args.ticker}")
    print(f"  Horizon: {args.horizon} trading days")
    print(f"  Context: {args.context} trading days")

    # --- Load data -----------------------------------------------------------
    df, is_synthetic = load_market_data(args.ticker, args.context + args.holdout, args.synthetic)

    if len(df) < 64:
        sys.exit(
            f"ERROR: need ≥ 64 data points for TimesFM (got {len(df)}).  "
            "Try a longer --context or check your ticker."
        )

    # --- Optional hold-out split for accuracy evaluation ---------------------
    accuracy: dict | None = None
    if args.holdout > 0:
        df_train = df.iloc[: -args.holdout]
        df_test  = df.iloc[-args.holdout :]
        holdout_actual = df_test["close"].values.astype(np.float32)
        print(f"\n  Hold-out: last {args.holdout} days reserved for evaluation "
              f"({df_test.index[0].date()} → {df_test.index[-1].date()})")
    else:
        df_train = df
        holdout_actual = None

    close_prices = df_train["close"].values.astype(np.float32)
    print(f"\n  Input series: {len(close_prices)} trading days  "
          f"(last close: {close_prices[-1]:,.2f})")

    # --- Run TimesFM forecast ------------------------------------------------
    point_fc, quant_fc = run_forecast(close_prices, horizon=args.horizon)

    # Clamp to positive (prices cannot go negative)
    point_fc = np.maximum(point_fc, 0.0)
    quant_fc = np.maximum(quant_fc, 0.0)

    # Build forecast date index (business days)
    last_date = df_train.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=args.horizon)

    # --- Accuracy evaluation (if hold-out requested) -------------------------
    if holdout_actual is not None:
        n_eval = min(args.horizon, len(holdout_actual))
        accuracy = compute_accuracy(holdout_actual[:n_eval], point_fc[:n_eval])
        print(f"\n  Accuracy vs hold-out ({n_eval} days):")
        print(f"    MAE:  {accuracy['mae']:,.2f} points")
        print(f"    RMSE: {accuracy['rmse']:,.2f} points")
        print(f"    MAPE: {accuracy['mape_pct']:.2f} %")
        print(f"    Directional accuracy: {accuracy['directional_accuracy_pct']:.1f} %")

    # --- Visualize -----------------------------------------------------------
    plot_forecast(
        df_train, forecast_dates, point_fc, quant_fc,
        args.ticker, is_synthetic, accuracy,
    )

    # --- Save CSV ------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)

    forecast_df = pd.DataFrame(
        {
            "date":           forecast_dates.strftime("%Y-%m-%d"),
            "point_forecast": np.round(point_fc, 2),
            "q10":            np.round(quant_fc[IDX_Q10], 2),
            "q20":            np.round(quant_fc[IDX_Q20], 2),
            "q80":            np.round(quant_fc[IDX_Q80], 2),
            "q90":            np.round(quant_fc[IDX_Q90], 2),
        }
    )
    csv_path = OUTPUT_DIR / "nifty50_forecast.csv"
    forecast_df.to_csv(csv_path, index=False)
    print(f"  ✅  Saved CSV:   {csv_path}")

    # --- Save JSON -----------------------------------------------------------
    output_json: dict = {
        "model": "TimesFM 1.0 (200M) PyTorch",
        "ticker": args.ticker,
        "data_source": "synthetic (GBM)" if is_synthetic else "Yahoo Finance (yfinance)",
        "input": {
            "context_days": len(close_prices),
            "last_observed_date": str(df_train.index[-1].date()),
            "last_close": round(float(close_prices[-1]), 2),
        },
        "forecast": {
            "horizon_days": args.horizon,
            "first_forecast_date": str(forecast_dates[0].date()),
            "last_forecast_date":  str(forecast_dates[-1].date()),
            "point": [round(float(v), 2) for v in point_fc],
            "q10":   [round(float(v), 2) for v in quant_fc[IDX_Q10]],
            "q90":   [round(float(v), 2) for v in quant_fc[IDX_Q90]],
        },
        "summary": {
            "forecast_mean":  round(float(point_fc.mean()), 2),
            "forecast_end":   round(float(point_fc[-1]), 2),
            "pct_change":     round(float((point_fc[-1] - close_prices[-1]) / close_prices[-1] * 100), 2),
            "pi80_width_end": round(float(quant_fc[IDX_Q90][-1] - quant_fc[IDX_Q10][-1]), 2),
        },
        "accuracy": accuracy,
        "notes": (
            "TimesFM is a general-purpose zero-shot foundation model. "
            "Financial markets are largely efficient — short-term price movements "
            "are near-random walk. These forecasts capture recent trend/momentum "
            "and should NOT be used as trading signals. "
            "See examples/market-forecast/README.md for a full capability assessment."
        ),
    }
    json_path = OUTPUT_DIR / "nifty50_forecast.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  ✅  Saved JSON:  {json_path}")

    # --- Summary print -------------------------------------------------------
    last_close = float(close_prices[-1])
    fc_end     = float(point_fc[-1])
    pct        = (fc_end - last_close) / last_close * 100

    print("\n" + "=" * 66)
    print("  FORECAST SUMMARY")
    print("=" * 66)
    print(f"\n  Last observed close : {last_close:>12,.2f}")
    print(f"  Forecast (Day 30)   : {fc_end:>12,.2f}  ({pct:+.2f} %)")
    print(f"  80 % PI (Day 30)    : "
          f"[{quant_fc[IDX_Q10][-1]:,.2f} — {quant_fc[IDX_Q90][-1]:,.2f}]")
    print(f"\n  ⚠️  Not a trading signal — see README.md for accuracy context.")
    print("=" * 66)


if __name__ == "__main__":
    main()
