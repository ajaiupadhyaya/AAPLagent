# AAPL Trading Agent Training Blueprint

## Objective

Build a model-driven AAPL trading agent that predicts **tradable edge** (not just direction), then converts edge to position sizing under explicit risk constraints.

---

## 1) Define strategy archetype first

Choose one primary archetype before modeling:

1. **Short-horizon swing** (2-10 trading days)
2. **Medium-horizon tactical** (2-8 weeks)
3. **Regime-adaptive hybrid** (switches horizon by volatility/macro regime)

The labeling, features, and execution policy should be consistent with this choice.

---

## 2) Data stack (minimum viable + advanced)

### Core market data

- AAPL OHLCV (daily + optional intraday)
- Benchmark context: SPY, QQQ
- Volatility proxy: VIX
- Rate proxy: US 10Y yield (`^TNX`)

### Enrichment data

- Earnings calendar and surprise history
- Corporate actions (splits/dividends)
- News sentiment (headline-level optional)
- Options surface proxies (IV rank, skew proxies if available)
- Macro calendar markers (CPI, FOMC, NFP)

### Fundamental snapshots

- Revenue growth, EPS growth
- Gross/operating margin trends
- Buyback activity, diluted shares trend
- Valuation proxies (P/E, EV/EBITDA, free cash flow yield)

### Data quality controls

- Timezone normalization
- Explicit point-in-time joins
- Missingness audit by feature
- Outlier handling policy (winsorize/cap/flag)
- Symbol continuity checks for index proxies

---

## 3) Feature engineering map

### Price/volume technical features

- Returns: 1d, 5d, 10d, 21d
- Rolling volatility: 10d/21d/63d annualized
- Momentum: RSI, MACD, stochastic
- Volatility envelopes: Bollinger width, Keltner width
- Trend state: moving-average spreads, slope features
- Volume state: z-score, volume trend, abnormal turnover

### Cross-asset and relative features

- AAPL excess return vs SPY/QQQ
- Beta to QQQ in rolling windows
- VIX level + change + percentile regime
- Yield level/change and interaction terms

### Event and seasonal features

- Earnings proximity (pre/post windows)
- Day-of-week, month-of-year
- Options-expiry-week flag
- Macro-event-day and next-day flags

### Risk-sensitive transforms

- Overnight return vs intraday return decomposition
- Gap size normalized by ATR
- Drawdown state (distance from local highs)
- Recent tail-risk proxy (left-tail quantile estimate)

---

## 4) Label design (critical)

Use at least two labels in research:

1. **Directional classification label**
   - Example: `1` if forward 5-day return > 2%, else `0`.
2. **Risk-adjusted regression label**
   - Example: forward return divided by trailing volatility.

Advanced option:

- Triple-barrier labeling (profit barrier, stop barrier, max holding period) for realistic entry/exit framing.

Avoid label leakage:

- Features must only use information available at decision timestamp.
- Ensure proper feature shifting for any rolling/effective-close logic.

---

## 5) Validation design (must be time-safe)

- Never use random K-fold for financial time series.
- Use **walk-forward** or **expanding-window** validation.
- Add **purging/embargo** when samples overlap by horizon.
- Evaluate across multiple regimes, not only full-period aggregate.

Recommended splits:

- Train: oldest 70-80%
- Validation: next 10-15%
- Test: most recent 10-15%

Then run rolling re-trains to mimic live operation.

---

## 6) Model stack progression

1. **Baseline**: logistic regression + XGBoost/LightGBM.
2. **Calibration**: isotonic or Platt scaling for probability reliability.
3. **Ensemble**: blend calibrated tree model + linear model.
4. **Optional sequence model**: only if incremental benefit over tree baseline is clear.

Hyperparameter process:

- Use Optuna for constrained search spaces.
- Optimize on a metric aligned with trading utility (e.g., net Sharpe proxy), not just AUC.

---

## 7) Decision policy (signal -> position)

Convert model output `p` to actions:

- `p < p_short`: reduce/flat (or short if strategy permits)
- `p in neutral band`: no trade
- `p > p_long`: long with size proportional to confidence and volatility target

Position sizing policy:

- Volatility-targeted sizing
- Cap max position and daily turnover
- Enforce exposure and concentration limits

---

## 8) Backtest requirements

Backtests must include:

- Commission + slippage + borrow assumptions (if shorting)
- Order timing assumptions (close-to-open, next-bar-open, etc.)
- Corporate actions treatment
- Latency-safe signal execution alignment

Report both raw and cost-adjusted metrics.

---

## 9) Evaluation metrics to track

### Prediction metrics

- ROC-AUC / PR-AUC
- Calibration error (Brier score)
- Precision/recall for trade-trigger zone

### Trading metrics

- CAGR
- Sharpe / Sortino
- Max drawdown
- Calmar ratio
- Profit factor
- Win rate + average win/loss
- Turnover and capacity proxy

### Stability metrics

- Regime-segmented performance
- Rolling 3/6/12-month Sharpe
- Feature importance drift
- Signal decay by holding horizon

---

## 10) Risk and governance framework

Required controls:

- Hard stop-loss and max time-in-trade
- Daily and weekly drawdown limits
- Exposure cut during high-volatility regimes
- Kill switch on data/API/model anomalies
- Model-version registry + reproducible artifacts

Monitoring:

- Data freshness checks
- Feature drift detection
- Prediction distribution drift
- Live-vs-backtest slippage gap monitoring

---

## 11) Suggested implementation milestones

### Phase 1 (now)

- Baseline data + technical features + binary model
- Basic walk-forward test with costs

### Phase 2

- Add cross-asset, event, and macro features
- Add probability calibration + confidence-based sizing

### Phase 3

- Add fundamentals and optional sentiment/options proxies
- Add regime classifier and strategy throttling

### Phase 4

- Paper trading integration, monitoring dashboard, and alerting

---

## 12) Non-negotiable safeguards

- No live trading before multi-regime out-of-sample success.
- No model updates without backtest + paper-trade checkpoint.
- No performance claims without costs and slippage.
- No unmanaged leverage.
