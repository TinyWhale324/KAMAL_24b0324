# HMM-Based Regime Detection & Dynamic Portfolio Optimization

> **Summer of Quant / Advanced Project**

A Python pipeline that detects hidden market regimes (Bull / Bear / Crisis) from NSE price data using a **Hidden Markov Model**, and dynamically re-allocates a multi-asset portfolio using **convex optimization (CVXPY)** , all validated with strict walk forward backtesting to prevent lookahead bias.

---

## Key Decisions

### Why 3 Regimes?
Financial literature and empirical observation suggest three dominant market states:
- **Bull**: Calm, upward-trending markets with low volatility.
- **Bear**: Declining markets with moderate volatility ,a sustained downturn.
- **Crisis**: Extreme, high-volatility periods (e.g., COVID crash of 2020, 2022 sell-off).

Three states offer a practical balance ,enough granularity to differentiate market environments without overfitting on noise.

### Why These Features?
| Feature | Window | Rationale |
|---------|--------|-----------|
| `mom_5d` | 5-day rolling mean return | Short-term momentum (weekly trend) |
| `mom_21d` | 21-day rolling mean return | Medium-term momentum (monthly trend) |
| `mom_63d` | 63-day rolling mean return | Long-term momentum (quarterly trend) |
| `vol_20d` | 20-day rolling std of returns | Recent realised volatility |
| `vol_60d` | 60-day rolling std of returns | Smoothed volatility trend |
| `vix` | India VIX (^INDIAVIX) | Market-implied fear gauge |

All features are **backward-looking** by construction (computed from past data only), which is critical for avoiding lookahead bias.

### Why These Assets?
| Asset | Ticker | Role |
|-------|--------|------|
| Nifty 50 | `^NSEI` | Indian equity market proxy |
| Gold Futures | `GC=F` | Safe-haven / inflation hedge |
| US Treasury Bond ETF | `TLT` | Fixed-income / flight-to-safety asset |

This covers three distinct asset classes (equities, commodities, bonds) with liquid, well-documented price histories on Yahoo Finance.

### Avoiding Lookahead Bias
The walk-forward validation engine guarantees no future data leakage:
1. At each rebalance point, the `StandardScaler` is fit **only on training data**.
2. The HMM is trained **only on past observations**.
3. The CVXPY optimizer uses **only training-period statistics** (mean returns, covariance).
4. Regime prediction for the current day uses the **same scaler and model** trained on the past.

---

## How to Run

### Prerequisites
- Python 3.9+
- A virtual environment is recommended

### Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install hmmlearn cvxpy yfinance pandas numpy matplotlib scipy scikit-learn seaborn
```

### Run the Pipeline
```bash
python regime_portfolio.py
```

### Output
All results are saved to the `output/` directory:
| File | Description |
|------|-------------|
| `regime_overlay.png` | Nifty 50 price chart with colour-coded regime bands |
| `equity_curves.png` | Equity curves: HMM strategy vs static benchmarks |
| `weight_evolution.png` | Stacked area chart of portfolio weight changes |
| `transition_matrix.png` | HMM state transition probability heatmap |
| `performance_summary.csv` | Sharpe, Sortino, Max Drawdown, Calmar, Turnover |

---

## Methodology

```
Daily Prices (yfinance)
    │
    ▼
Feature Engineering ──► Momentum (5d, 21d, 63d) + Volatility (20d, 60d) + VIX
    │
    ▼
Walk-Forward Loop ─────────────────────────────────────
    │  For each monthly rebalance:                     │
    │    1. Fit StandardScaler on [0, t)               │
    │    2. Fit GaussianHMM on [0, t)                  │
    │    3. Predict regime at t                        │
    │    4. Optimize weights via CVXPY for regime      │
    │    5. Apply weights for [t, t+21)                │
    ────────────────────────────────────────────────────
    │
    ▼
Backtesting ──► Apply transaction costs (7 bps / rebalance)
    │
    ▼
Performance Metrics ──► Sharpe, Sortino, Max DD, Calmar, Turnover
    │
    ▼
Compare vs. Static 60/40 and Equal-Weight benchmarks
```

### Optimization Objectives by Regime
| Regime | CVXPY Objective | Rationale |
|--------|----------------|-----------|
| **Bull** | Maximise risk-adjusted return (`max μᵀw − λwᵀΣw`) | Capture upside aggressively |
| **Bear** | Minimise variance with return floor | Defensive but not fully risk-off |
| **Crisis** | Pure minimum variance (`min wᵀΣw`) | Capital preservation |

All optimizations are **long-only** with weights summing to 1.

---

## Performance Metrics

| Metric | Definition |
|--------|-----------|
| **Sharpe Ratio** | Annualised return / annualised volatility |
| **Sortino Ratio** | Annualised return / downside deviation |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | Annualised return / |max drawdown| |
| **Turnover** | Average weight change per rebalance |

Results are reported **with and without** transaction costs (7 bps per rebalance).

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `yfinance` | Market data download |
| `pandas` / `numpy` | Data manipulation |
| `hmmlearn` | Gaussian Hidden Markov Model |
| `scikit-learn` | Feature scaling (StandardScaler) |
| `cvxpy` | Convex portfolio optimization |
| `matplotlib` | Visualisation |
| `scipy` | Numerical utilities |

---

## Reproducing Results

```bash
git clone <repo-url>
#Navigate to root folder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or install packages manually
python regime_portfolio.py
# Results → output/ directory
```

> **Note**: Results may vary slightly across runs due to HMM random initialisation, but the overall regime structure and performance ranking should remain consistent.
