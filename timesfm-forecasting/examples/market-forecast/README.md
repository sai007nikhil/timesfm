# TimesFM Market Forecast — Nifty 50 Example

This example demonstrates how to use **TimesFM** for **real-time market data
prediction**, using the **Nifty 50** index as the reference series.

---

## How it works

```
yfinance (live data)
      │
      ▼
1 year of daily Nifty 50 closes  ──►  TimesFM 1.0 (zero-shot)  ──►  30-day forecast
                                                                       + 80 % PI bands
```

`forecast_nifty50.py` fetches data up to the most recent trading day, feeds the
closing-price history to TimesFM as a single univariate context, then plots the
forecast with calibrated 60 % and 80 % prediction intervals.

---

## Quick start

```bash
# 1 — install optional market-data dependency
pip install yfinance

# 2 — run the example (downloads Nifty 50 from Yahoo Finance)
python forecast_nifty50.py

# 3 — evaluate accuracy on the last 30 days (hold-out test)
python forecast_nifty50.py --holdout 30

# 4 — run without network (synthetic GBM data)
python forecast_nifty50.py --synthetic

# 5 — other Indian / global indices
python forecast_nifty50.py --ticker "^NSEBANK"   # Bank Nifty
python forecast_nifty50.py --ticker "^BSESN"     # BSE Sensex
python forecast_nifty50.py --ticker "^GSPC"      # S&P 500
python forecast_nifty50.py --ticker "^DJI"       # Dow Jones
python forecast_nifty50.py --ticker "AAPL"       # Individual stocks work too
```

---

## Output files

| File | Description |
|------|-------------|
| `output/nifty50_forecast.png` | 2-panel chart: price history + forecast + PI bands, uncertainty growth |
| `output/nifty50_forecast.csv` | Point forecast + q10/q20/q80/q90 per trading day |
| `output/nifty50_forecast.json` | Structured results, accuracy metrics, metadata |

---

## How well does TimesFM work on market data?

### Short answer

TimesFM is **useful for capturing trend and momentum** over a multi-week horizon.
It is **not a substitute for quantitative trading systems** and should not be used
as a standalone trading signal.

### What TimesFM can do

| Capability | Notes |
|------------|-------|
| **Trend extrapolation** | If the index has been rising steadily, TimesFM will usually project continuation — useful as a *baseline*. |
| **Volatility-aware intervals** | The 80 % prediction interval widens over the horizon, reflecting growing uncertainty. Use these bands for risk-range estimation. |
| **Momentum patterns** | Short cycles and momentum patterns visible in the recent context are captured by the attention mechanism. |
| **Batch forecasting** | Hundreds of stocks/indices in a single `model.forecast()` call — great for screening. |
| **No training required** | Zero-shot inference on any price series, any market, any currency. |

### What TimesFM cannot do

| Limitation | Why |
|------------|-----|
| **Predict black-swan events** | Crashes, circuit breakers, news shocks are outside any historical pattern. |
| **Beat random-walk in short-term** | Daily returns are near-random (EMH). MAE on a 1-day horizon is typically larger than a naive *"tomorrow = today"* baseline. |
| **Model microstructure** | Order-book dynamics, bid-ask spread, intraday patterns — TimesFM only sees daily close prices. |
| **Understand fundamentals** | P/E ratios, earnings surprises, RBI policy — none of this enters the model. |
| **Causal inference** | TimesFM identifies correlational patterns, not causal drivers. |

### Realistic accuracy benchmarks (Nifty 50, 30-day horizon)

These figures are indicative; actual results vary with market regime.

| Metric | Typical range | Interpretation |
|--------|---------------|----------------|
| MAPE (30 days) | 2 – 6 % | Good for trend; poor for turning points |
| Directional accuracy | 50 – 60 % | Marginally better than coin flip |
| 80 % PI coverage | ~75 – 85 % | Well-calibrated uncertainty |

> **Key insight**: A MAPE of 3 % sounds small, but on Nifty 50 at 22 000 that is
> ±660 points — a range where the market can easily land anywhere.  The value is in
> the **direction** and **uncertainty range**, not in the precise point estimate.

### Recommended usage patterns

1. **Risk range estimation** — use the 80 % PI as a plausible operating range for
   the next month when stress-testing portfolios.

2. **Trend filtering** — if TimesFM forecasts a rising trend AND other indicators
   agree, higher conviction entry.  If they diverge, add caution.

3. **Multi-series screening** — forecast 50 Nifty constituents at once; rank by
   predicted return / risk ratio (point / PI width) to build a candidate list.

4. **Anomaly detection** — compare actual price to the forecast PI; values outside
   the 90 % CI on consecutive days signal a regime break.

5. **Do NOT** treat the point forecast as a price target for individual trades.

---

## Improving accuracy with covariates (TimesFM 2.5 + XReg)

For market data, you can add exogenous variables with `forecast_with_covariates()`:

```python
# Dynamic numerical covariates (must span context + horizon)
dynamic_numerical_covariates = {
    "vix": vix_series,           # implied volatility index
    "rsi": rsi_series,           # technical RSI indicator
    "volume": volume_series,     # trading volume (log-normalise)
}

# Dynamic categorical covariates
dynamic_categorical_covariates = {
    "day_of_week": dow_series,   # 0=Mon … 4=Fri
    "expiry_week": expiry_flag,  # 1 on F&O expiry weeks
}

point_fc, quant_fc = model.forecast_with_covariates(
    inputs=[close_prices],
    dynamic_numerical_covariates=dynamic_numerical_covariates,
    xreg_mode="xreg + timesfm",
)
```

> ⚠️ Dynamic covariates require **known future values** over the forecast horizon.
> VIX and volume cannot be known in advance — you must supply a forecast or
> assume they remain constant.  Only use covariates whose future values you can
> reliably estimate.

---

## Data sources

| Source | Ticker format | Latency | Cost |
|--------|--------------|---------|------|
| Yahoo Finance (`yfinance`) | `^NSEI`, `^GSPC` … | 15–20 min | Free |
| NSE India official API | — | Real-time | Free (registration) |
| Zerodha Kite Connect | Symbol names | Real-time | Paid API |
| Alpha Vantage | `BSE:NIFTY` | 1–5 min | Free tier available |

The example uses `yfinance` for simplicity.  For sub-minute intraday data,
replace the `fetch_nifty_data()` function with your broker API.

---

## Limitations disclaimer

> This example is for **educational and research purposes only**.
> TimesFM forecasts are statistical extrapolations — they are **not financial
> advice and must not be used as trading signals**.  Past performance of a
> model on historical data does not guarantee future accuracy.
> Always consult a qualified financial advisor before making investment decisions.
