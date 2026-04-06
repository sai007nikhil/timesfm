# Market Data Guide — Using TimesFM with Financial Time Series

This reference covers how to prepare, feed, and interpret TimesFM forecasts when
working with market data (equity indices, individual stocks, FX rates, commodities).

---

## Table of contents

1. [Data preparation](#1-data-preparation)
2. [Model configuration for market data](#2-model-configuration)
3. [Realistic accuracy expectations](#3-realistic-accuracy-expectations)
4. [Prediction intervals as risk ranges](#4-prediction-intervals-as-risk-ranges)
5. [Working with covariates](#5-working-with-covariates)
6. [Batch forecasting a universe of stocks](#6-batch-forecasting)
7. [Anomaly detection on price series](#7-anomaly-detection)
8. [Common pitfalls](#8-common-pitfalls)

---

## 1. Data preparation

### Fetching real-time data with yfinance

```python
import yfinance as yf
import numpy as np
import pandas as pd

ticker = "^NSEI"   # Nifty 50
df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)

# Drop NaN rows (weekends, public holidays already absent in daily data)
close = df["Close"].dropna().values.astype(np.float32)
```

### Handling non-trading days

Daily OHLCV data from Yahoo Finance already excludes weekends and holidays —
**do not forward-fill or interpolate**.  Pass the series as-is.

```python
# ✅ Correct — just use the business-day close series
inputs = [close]

# ❌ Incorrect — reindexing to calendar days introduces artificial flat segments
# df_reindexed = df.reindex(pd.date_range(...))   # DON'T do this
```

### Minimum context length

TimesFM requires **≥ 32 data points**.  For daily market data:

| Context | Trading days | Calendar time |
|---------|-------------|---------------|
| Minimum | 32 | ~6.5 weeks |
| Recommended | 252 | ~1 year |
| Long-range | 1 024 | ~4 years |

### Raw prices vs. log-returns

TimesFM works best on **raw close prices** with `normalize_inputs=True` (TimesFM 2.5)
or without any preprocessing (TimesFM 1.0).  Do **not** compute log-returns
before passing to TimesFM — the model would then forecast returns, which you
would need to cumulate back to prices, compounding errors.

```python
# ✅ Use raw close prices
inputs = [close_prices]

# ❌ Don't transform to returns first
# log_ret = np.diff(np.log(close_prices))   # TimesFM forecasts returns → hard to use
```

---

## 2. Model configuration

### TimesFM 1.0 (current stable, works with raw prices)

```python
import timesfm

HORIZON = 30   # trading days

hparams = timesfm.TimesFmHparams(horizon_len=HORIZON)
checkpoint = timesfm.TimesFmCheckpoint(
    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
)
model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

point_fc, quant_fc = model.forecast(
    [close_prices],
    freq=[0],   # Required for TimesFM 1.0; daily data → freq=0
)
# point_fc.shape  == (1, HORIZON)
# quant_fc.shape  == (1, HORIZON, 10)
```

### TimesFM 2.5 (longer context, normalize_inputs)

```python
import torch, timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=1024,
    max_horizon=30,
    normalize_inputs=True,      # recommended for price data
    infer_is_positive=True,     # prices are always positive
    fix_quantile_crossing=True,
))

point_fc, quant_fc = model.forecast(horizon=30, inputs=[close_prices])
```

### Key parameters for financial data

| Parameter | Recommended value | Reason |
|-----------|-----------------|--------|
| `normalize_inputs` | `True` | Prices span orders of magnitude across tickers |
| `infer_is_positive` | `True` | Stock / index prices cannot go negative |
| `fix_quantile_crossing` | `True` | Ensures q10 ≤ q20 ≤ … ≤ q90 |
| `freq` (1.0 only) | `0` | Default / daily; required for TimesFM 1.0 |

---

## 3. Realistic accuracy expectations

Financial markets are hard to forecast.  The **efficient market hypothesis**
states that all publicly available information is already priced in, leaving only
noise for short-horizon point forecasts.

### What TimesFM captures

- **Trend / momentum** in the recent context window (5–20 day patterns)
- **Weekly seasonality** (e.g., Friday-effect, post-expiry patterns) if persistent
- **Gradual mean reversion** in relatively stable regimes

### What TimesFM cannot capture

- Sudden news shocks (earnings surprises, central bank policy changes)
- Structural breaks (regime changes from bull to bear market)
- Microstructure effects (intraday, bid-ask dynamics)
- Fundamental valuation (P/E expansion, macro growth)

### Indicative benchmark numbers (Nifty 50 daily close, 30-day horizon)

| Metric | Naive (last value) | TimesFM |
|--------|--------------------|---------|
| MAPE | ~3.5 % | ~2.5 – 4 % |
| Directional accuracy | 50 % | 52 – 58 % |
| 80 % PI coverage | — | ~75 – 85 % |

> TimesFM sometimes beats the naive baseline on MAPE thanks to trend capture,
> but the advantage shrinks at longer horizons and disappears in choppy markets.

---

## 4. Prediction intervals as risk ranges

The most actionable output from TimesFM for market data is the **prediction
interval**, not the point forecast.

```python
# Index convention: 0=mean, 1=q10, 2=q20 ... 9=q90
IDX_Q10, IDX_Q90 = 1, 9

q10 = quant_fc[0, :, IDX_Q10]   # lower bound of 80 % PI
q90 = quant_fc[0, :, IDX_Q90]   # upper bound of 80 % PI

print(f"Expected range in 30 days: [{q10[-1]:,.0f} — {q90[-1]:,.0f}]")
```

**Use cases for PI in finance**:

- **Portfolio stress-testing** — assume the 10th percentile scenario and size
  positions so the portfolio survives it.
- **Options sizing** — the PI width is a proxy for implied volatility over the
  forecast horizon; compare it to options market pricing.
- **Stop-loss placement** — place stops below the q10 level (price falling below
  the lower 80 % PI is statistically unusual).

---

## 5. Working with covariates

TimesFM 2.5 supports exogenous variables through `forecast_with_covariates()`.
Useful financial covariates:

| Covariate | Type | Notes |
|-----------|------|-------|
| VIX / India VIX | dynamic numerical | Must forecast future VIX or assume constant |
| RSI / MACD | dynamic numerical | Technical indicators from the same history |
| F&O expiry flag | dynamic categorical | 1 on weekly/monthly expiry dates |
| Day of week | dynamic categorical | Monday–Friday (0–4) |
| Macro regime | static categorical | "bull" / "bear" / "neutral" |

> ⚠️ **Future covariates must be known for the full forecast horizon.**
> Do not use covariates whose future values you cannot reliably predict.

```python
# Build day-of-week arrays spanning context + horizon
n_context = len(close_prices)
n_total = n_context + HORIZON

dow_full = np.arange(n_total) % 5   # 0=Mon, 4=Fri (simplified)

point_fc, quant_fc = model.forecast_with_covariates(
    inputs=[close_prices],
    dynamic_categorical_covariates={"day_of_week": [dow_full]},
    xreg_mode="xreg + timesfm",
)
```

---

## 6. Batch forecasting

Forecast a full universe of Nifty 50 constituents in one call:

```python
import yfinance as yf

# Download all 50 constituents (ticker list from NSE)
nifty50_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", ...]   # full list

df = yf.download(nifty50_tickers, period="1y", progress=False, auto_adjust=True)["Close"]
df = df.dropna()

inputs = [df[col].values.astype(np.float32) for col in df.columns]

# Single model.forecast() call for all 50 series
point_fc, quant_fc = model.forecast(horizon=30, inputs=inputs)
# point_fc.shape == (50, 30)

# Rank by predicted return / uncertainty ratio
last_prices = np.array([inp[-1] for inp in inputs])
pred_return = (point_fc[:, -1] - last_prices) / last_prices * 100
pi_width    = quant_fc[:, -1, IDX_Q90] - quant_fc[:, -1, IDX_Q10]
risk_adj    = pred_return / (pi_width / last_prices * 100)

ranking = pd.Series(risk_adj, index=df.columns).sort_values(ascending=False)
print(ranking.head(10))
```

For very large universes (> 500 series), process in chunks of 50 to avoid OOM:

```python
CHUNK = 50
results = []
for i in range(0, len(inputs), CHUNK):
    p, q = model.forecast(horizon=30, inputs=inputs[i:i+CHUNK])
    results.append((p, q))
```

---

## 7. Anomaly detection

Use the quantile PI to flag unusual price moves on live data:

```python
def check_for_anomaly(actual_price: float, q10: float, q90: float) -> str:
    """Classify a new observation against the forecast interval."""
    if actual_price < q10 or actual_price > q90:
        return "CRITICAL"   # outside 80 % PI
    q20 = q10 + (q90 - q10) * (2 / 8)   # approximate q20
    q80 = q10 + (q90 - q10) * (6 / 8)   # approximate q80
    if actual_price < q20 or actual_price > q80:
        return "WARNING"    # outside 60 % PI
    return "NORMAL"

# On each new trading day:
day_idx = 0   # first forecast day
status = check_for_anomaly(
    actual_price=actual_close,
    q10=quant_fc[0, day_idx, IDX_Q10],
    q90=quant_fc[0, day_idx, IDX_Q90],
)
```

---

## 8. Common pitfalls

| Pitfall | Fix |
|---------|-----|
| Passing log-returns instead of prices | Pass raw close prices; TimesFM handles scale internally |
| Calendar-day index with NaNs for weekends | Use trading-day (business-day) index only |
| `freq=[0]` omitted for TimesFM 1.0 | Always pass `freq=[0]` for daily data with TimesFM 1.0 |
| `infer_is_positive=False` for price data | Set `True`; prices cannot be negative |
| Treating point forecast as price target | Use it as the *centre* of a wide uncertainty range |
| Future covariates with unknown values | Only use covariates you can reliably forecast or set constant |
| Context shorter than 32 points | Ensure `len(inputs[i]) >= 32` before calling `model.forecast()` |
| Comparing to calendar-time competitors | TimesFM forecasts trading days — align evaluation dates accordingly |
