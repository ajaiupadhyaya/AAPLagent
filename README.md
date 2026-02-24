# AAPL Trading Agent

This repository is structured to build a **research-first, risk-aware, model-driven trading agent** focused on AAPL.

## 1) Quick start

### Create and activate virtual environment

```bash
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

### Fetch data and run baseline training

```bash
python scripts/fetch_market_data.py
python scripts/train_baseline.py
python scripts/train_intraday_baseline.py
python scripts/generate_model_proof.py
python scripts/train_intraday_enhanced.py
python scripts/generate_improvement_report.py
python scripts/train_intraday_round2.py
python scripts/generate_round2_report.py
```

This repository now includes a user-aligned profile for:

- Hybrid regime-adaptive strategy
- Long/short directionality
- Intraday execution style
- Balanced risk profile

Config: `configs/intraday_hybrid.yaml`

## 2) What the agent should use (beyond price)

A robust AAPL agent should combine:

- **Technical structure**: returns, momentum (RSI/MACD), trend persistence, volatility regime, ATR, volume shocks.
- **Cross-asset context**: SPY/QQQ relative strength, VIX level/change, yields (`^TNX`), sector-beta context.
- **Event context**: earnings windows, post-earnings drift periods, macro event days, options expiry proximity.
- **Fundamental context**: revenue/earnings growth trends, margins, buyback cadence, valuation regime.
- **Risk & microstructure proxies**: gap risk, overnight/intraday decomposition, liquidity proxies (spread/volume).
- **Execution-aware variables**: slippage assumptions, market impact estimates, broker fill quality.

## 3) End-to-end build order

1. **Data layer**: ingest and version all market + context + event + fundamentals data.
2. **Feature layer**: generate lagged, normalized, leakage-safe features.
3. **Labeling layer**: define objective(s): direction, risk-adjusted return, drawdown-penalized score.
4. **Modeling layer**: baseline tree model -> tuned ensemble -> probability calibration.
5. **Backtesting layer**: walk-forward with transaction costs and strict no-lookahead checks.
6. **Risk layer**: stop framework, volatility targeting, position limits, regime throttles.
7. **Paper-trade layer**: broker integration, logging, drift monitoring, fail-safe handling.

## 4) Minimum production risk controls

- Max position size and max gross exposure caps.
- Daily loss limit and kill-switch.
- Confidence threshold for trade entry.
- Hard stop-loss and time-based exits.
- Trade cooldown around high-risk events.
- Model drift alerts (feature drift + performance decay).

## 5) Immediate next steps

- Expand from baseline technical features to a full feature store.
- Add purged walk-forward CV and probability calibration.
- Add a portfolio simulator with realistic costs and partial fills.
- Add paper-trading loop and daily health report.

See `docs/training_blueprint.md` for detailed implementation guidance.
