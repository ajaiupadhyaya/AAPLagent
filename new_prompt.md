You are a senior quantitative engineer and ML researcher. Your job is to evolve my existing AAPL analysis repo into a **production-grade, institutional-level research and trading intelligence system**.

You must:

* Preserve what already works
* Refactor where needed
* Add state-of-the-art modeling
* Enforce rigorous quant methodology
* Automate the full pipeline

The end result should feel like a **mini hedge-fund research stack**, not a hobby project.

---

# 🎯 HIGH-LEVEL OBJECTIVES

Transform the repo into a system that can:

1. Produce robust forward return predictions for AAPL
2. Model regime changes and volatility structure
3. Avoid data leakage and overfitting
4. Continuously retrain and evaluate itself
5. Generate professional-grade research reports
6. Support future multi-asset expansion
7. Run fully automated end-to-end

---

# 🧱 PHASE 1 — DATA PIPELINE HARDENING

## 1.1 Data validation layer

Create:

```
src/aapl_agent/data_validation.py
```

Add checks for:

* missing timestamps
* duplicate rows
* split/dividend adjustments
* intraday session gaps
* timezone normalization
* outlier detection (z-score + MAD)

Fail loudly if data integrity is compromised.

---

## 1.2 Feature store architecture

Refactor features into a **feature registry pattern**.

Create:

```
src/aapl_agent/feature_store/
    registry.py
    price_features.py
    volatility_features.py
    microstructure_features.py
    macro_features.py
```

Requirements:

* Each feature is a pure function
* Deterministic outputs
* Cached to disk (parquet)
* Versioned by hash
* No lookahead leakage

---

## 1.3 Expand feature set (institutional level)

Implement the following categories:

### Price/technical

* multi-horizon returns (1,5,10,20,60)
* rolling VWAP distance
* realized volatility (Parkinson, Garman-Klass)
* ATR variants
* trend slope via rolling regression
* volume-weighted momentum

### Volatility surface proxies

* VIX term structure features
* realized vs implied spread
* volatility of volatility
* regime volatility percentile

### Market context

Using SPY, QQQ, TNX, VIX:

* rolling beta to SPY
* relative strength vs QQQ
* rate sensitivity to TNX
* correlation regime features
* cross-asset momentum

### Intraday microstructure (IMPORTANT)

From 5-minute data:

* intraday seasonality encoding
* opening range breakout features
* volume imbalance proxy
* intraday volatility buckets
* time-since-open encoding
* lunch hour dummy

---

# 🧠 PHASE 2 — LABEL ENGINEERING (CRITICAL)

Refactor labels into a modular system.

Create:

```
src/aapl_agent/labels/
    forward_returns.py
    triple_barrier.py
    meta_labels.py
```

Implement:

## 2.1 Multi-horizon forward returns

* 1d
* 5d
* 10d
* intraday horizons

---

## 2.2 Triple barrier labeling (Lopez de P



