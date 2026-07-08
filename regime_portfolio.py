#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
 HMM-Based Regime Detection & Dynamic Portfolio Optimization
 Summer of Quant — Advanced Project
══════════════════════════════════════════════════════════════════════
Pipeline: Data → Features → Regime Detection → Optimization → Backtest

Run:  python regime_portfolio.py
"""

import warnings
warnings.filterwarnings('ignore')

# ── SSL workaround for sandboxed Linux environments ──
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
import os, sys

# ── Configuration ────────────────────────────────────────────────
ASSETS = {'^NSEI': 'Nifty 50', 'GC=F': 'Gold', 'TLT': 'Bonds (TLT)'}
VIX_TICKER  = '^INDIAVIX'
START_DATE  = '2010-01-01'
END_DATE    = '2024-12-31'

N_REGIMES           = 3
INITIAL_TRAIN_DAYS  = 504      # ~2 years
REBALANCE_FREQ      = 21       # ~monthly
TC_BPS              = 7        # transaction cost in basis points
HMM_ITER            = 200
HMM_RESTARTS        = 10

REGIME_COLORS = {'Bull': '#2ecc71', 'Bear': '#e74c3c', 'Crisis': '#8e44ad', 'Unknown': '#95a5a6'}
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════
#  PHASE 1 — DATA INGESTION
# ═════════════════════════════════════════════════════════════════

def fetch_data():
    """Download daily price data for assets and India VIX via yfinance."""
    print("=" * 60)
    print(" PHASE 1: DATA INGESTION")
    print("=" * 60)

    prices = pd.DataFrame()
    for ticker, name in ASSETS.items():
        print(f"  Downloading {name} ({ticker})...")
        raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        col = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
        # yfinance may return MultiIndex columns; flatten
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        prices[ticker] = col

    print(f"  Downloading India VIX ({VIX_TICKER})...")
    vix_raw = yf.download(VIX_TICKER, start=START_DATE, end=END_DATE, progress=False)
    vix_col = vix_raw['Close'] if 'Close' in vix_raw.columns else vix_raw.iloc[:, 0]
    if isinstance(vix_col, pd.DataFrame):
        vix_col = vix_col.iloc[:, 0]
    vix = vix_col.squeeze()

    # Align and clean
    prices = prices.ffill().dropna()
    vix = vix.reindex(prices.index).ffill()
    returns = prices.pct_change().dropna()
    common = returns.index.intersection(vix.dropna().index)
    returns, prices, vix = returns.loc[common], prices.loc[common], vix.loc[common]

    print(f"  Date range : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"  Trading days: {len(returns)}")
    return prices, returns, vix


# ═════════════════════════════════════════════════════════════════
#  PHASE 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════

def compute_features(returns, vix, primary='^NSEI'):
    """Backward-looking momentum & volatility features from primary asset."""
    print("\n" + "=" * 60)
    print(" PHASE 2: FEATURE ENGINEERING")
    print("=" * 60)

    r = returns[primary]
    feat = pd.DataFrame(index=returns.index)

    # Momentum (rolling mean return)
    feat['mom_5d']  = r.rolling(5).mean()
    feat['mom_21d'] = r.rolling(21).mean()
    feat['mom_63d'] = r.rolling(63).mean()

    # Volatility (rolling std)
    feat['vol_20d'] = r.rolling(20).std()
    feat['vol_60d'] = r.rolling(60).std()

    # India VIX (observable — no lookahead)
    feat['vix'] = vix

    feat.dropna(inplace=True)
    print(f"  Features   : {list(feat.columns)}")
    print(f"  Shape      : {feat.shape}")
    return feat


# ═════════════════════════════════════════════════════════════════
#  PHASE 3 — HMM REGIME CLASSIFIER
# ═════════════════════════════════════════════════════════════════

def fit_hmm(X_scaled):
    """Fit GaussianHMM with multiple random restarts; return best model."""
    best, best_score = None, -np.inf
    for seed in range(HMM_RESTARTS):
        m = GaussianHMM(n_components=N_REGIMES, covariance_type='full',
                        n_iter=HMM_ITER, random_state=seed, tol=1e-4)
        try:
            m.fit(X_scaled)
            s = m.score(X_scaled)
            if s > best_score:
                best, best_score = m, s
        except Exception:
            continue
    return best


def map_states_to_regimes(model, X_scaled, returns_aligned, primary='^NSEI'):
    """Assign Bull/Bear/Crisis labels by analysing each state's statistics."""
    states = model.predict(X_scaled)
    stats = {}
    for s in range(model.n_components):
        mask = states == s
        r = returns_aligned[primary].values[mask]
        stats[s] = {'mean': np.mean(r), 'vol': np.std(r)}

    crisis = max(stats, key=lambda s: stats[s]['vol'])
    rest   = [s for s in stats if s != crisis]
    bull   = max(rest, key=lambda s: stats[s]['mean'])
    bear   = [s for s in rest if s != bull][0]

    smap = {bull: 'Bull', bear: 'Bear', crisis: 'Crisis'}
    labels = pd.Series([smap[s] for s in states], index=returns_aligned.index)
    return labels, smap


# ═════════════════════════════════════════════════════════════════
#  PHASE 5 — PORTFOLIO OPTIMIZATION (CVXPY)
# ═════════════════════════════════════════════════════════════════

def optimize_weights(train_returns, regime_labels, regime, asset_list):
    """
    Solve for optimal weights given the current regime.
      Bull  → maximise risk-adjusted return
      Bear  → minimum variance with modest return floor
      Crisis→ pure minimum variance
    Uses ONLY training-period statistics.
    """
    n = len(asset_list)
    mask = regime_labels == regime
    rr = train_returns.loc[mask]

    if len(rr) < 30:                       # too few observations
        return np.ones(n) / n

    mu    = rr.mean().values * 252          # annualised expected return
    Sigma = rr.cov().values * 252 + np.eye(n) * 1e-6   # regularised cov

    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0]

    try:
        if regime == 'Bull':
            obj = cp.Maximize(mu @ w - 1.0 * cp.quad_form(w, Sigma))
        elif regime == 'Bear':
            obj = cp.Minimize(cp.quad_form(w, Sigma))
            if mu.mean() > 0:
                cons.append(mu @ w >= 0.5 * mu.mean())
        else:   # Crisis
            obj = cp.Minimize(cp.quad_form(w, Sigma))

        cp.Problem(obj, cons).solve(solver=cp.SCS, verbose=False)
        if w.value is not None:
            wv = np.maximum(w.value, 0)
            return wv / wv.sum()
    except Exception:
        pass
    return np.ones(n) / n


# ═════════════════════════════════════════════════════════════════
#  PHASE 4 — WALK-FORWARD VALIDATION
# ═════════════════════════════════════════════════════════════════

def walk_forward(features, returns, primary='^NSEI'):
    """
    Expanding-window walk-forward backtest.
    At each rebalance point t:
      • fit scaler + HMM on data [0, t) only
      • predict regime at t
      • optimise weights for next period using training stats
    """
    print("\n" + "=" * 60)
    print(" PHASE 4: WALK-FORWARD VALIDATION")
    print("=" * 60)

    common = features.index.intersection(returns.index)
    features, returns = features.loc[common], returns.loc[common]
    n = len(features)
    assets = list(returns.columns)
    na = len(assets)

    regime_s  = pd.Series('Unknown', index=features.index)
    weights_df = pd.DataFrame(np.nan, index=features.index, columns=assets)
    rebal_dates = []

    eq_w = np.ones(na) / na
    cur_w = eq_w.copy()

    # Fill initial training window with equal weights
    weights_df.iloc[:INITIAL_TRAIN_DAYS] = eq_w

    count = 0
    for t in range(INITIAL_TRAIN_DAYS, n, REBALANCE_FREQ):
        # ── train on [0, t) ──
        train_feat = features.iloc[:t].values
        train_ret  = returns.iloc[:t]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_feat)

        model = fit_hmm(X_train)
        if model is None:
            end = min(t + REBALANCE_FREQ, n)
            weights_df.iloc[t:end] = cur_w
            continue

        labels_train, smap = map_states_to_regimes(model, X_train, train_ret, primary)

        # ── predict regime at rebalance date ──
        x_now = scaler.transform(features.iloc[t-1:t].values)
        state_now = model.predict(x_now)[0]
        regime_now = smap.get(state_now, 'Bull')

        cur_w = optimize_weights(train_ret, labels_train, regime_now, assets)

        end = min(t + REBALANCE_FREQ, n)
        regime_s.iloc[t:end] = regime_now
        weights_df.iloc[t:end] = cur_w
        rebal_dates.append(features.index[t])
        count += 1

    weights_df = weights_df.ffill().bfill()
    print(f"  Rebalances : {count}")
    return regime_s, weights_df, rebal_dates


# ═════════════════════════════════════════════════════════════════
#  PHASE 6 — BACKTESTING & METRICS
# ═════════════════════════════════════════════════════════════════

def compute_metrics(ret_series, name, rf=0.0):
    """Compute Sharpe, Sortino, MaxDD, Calmar for a return series."""
    r = ret_series.dropna()
    ann_ret  = r.mean() * 252
    ann_vol  = r.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    down_vol = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-9
    sortino  = ann_ret / down_vol
    cum      = (1 + r).cumprod()
    dd       = cum / cum.cummax() - 1
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
    return {'Strategy': name, 'Ann. Return': f"{ann_ret:.2%}", 'Ann. Vol': f"{ann_vol:.2%}",
            'Sharpe': round(sharpe, 3), 'Sortino': round(sortino, 3),
            'Max Drawdown': f"{max_dd:.2%}", 'Calmar': round(calmar, 3)}


def backtest(returns, weights_df, rebal_dates):
    """Run backtest for dynamic strategy and benchmarks."""
    print("\n" + "=" * 60)
    print(" PHASE 6: BACKTESTING")
    print("=" * 60)

    common = returns.index.intersection(weights_df.index)
    ret = returns.loc[common]
    wts = weights_df.loc[common].values
    na  = ret.shape[1]

    # Dynamic strategy (gross)
    port_gross = (ret.values * wts).sum(axis=1)

    # Dynamic strategy (net of TC)
    port_net = port_gross.copy()
    tc = TC_BPS / 10_000
    rebal_set = set(rebal_dates)
    prev_w = wts[0]
    turnover_total = 0.0
    for i in range(1, len(ret)):
        if ret.index[i] in rebal_set:
            to = np.sum(np.abs(wts[i] - prev_w))
            turnover_total += to
            port_net[i] -= to * tc
            prev_w = wts[i]

    # Static 60/40 (60 equity, 20 gold, 20 bonds)
    w6040 = np.array([0.60, 0.20, 0.20])
    port_6040 = (ret.values * w6040).sum(axis=1)

    # Equal weight
    weq = np.ones(na) / na
    port_eq = (ret.values * weq).sum(axis=1)

    strats = {
        'HMM Strategy (Gross)': pd.Series(port_gross, index=common),
        'HMM Strategy (Net)':   pd.Series(port_net,   index=common),
        'Static 60/40':         pd.Series(port_6040,  index=common),
        'Equal Weight':         pd.Series(port_eq,     index=common),
    }

    equities = {k: (1 + v).cumprod() for k, v in strats.items()}

    # Metrics table
    rows = [compute_metrics(v, k) for k, v in strats.items()]
    # Add turnover for dynamic
    avg_turnover = turnover_total / max(len(rebal_dates), 1)
    rows[0]['Turnover/Rebal'] = f"{avg_turnover:.2%}"
    rows[1]['Turnover/Rebal'] = f"{avg_turnover:.2%}"
    metrics_df = pd.DataFrame(rows)

    print("\n  Performance Summary")
    print("  " + "─" * 56)
    print(metrics_df.to_string(index=False))

    return strats, equities, metrics_df


# ═════════════════════════════════════════════════════════════════
#  PHASE 7 — VISUALISATION
# ═════════════════════════════════════════════════════════════════

def plot_all(prices, regime_series, equities, weights_df, metrics_df):
    """Generate all required plots and save to output/."""
    print("\n" + "=" * 60)
    print(" PHASE 7: VISUALISATION")
    print("=" * 60)

    primary = '^NSEI'
    idx = prices.index.intersection(regime_series.index)

    # ── 1. Price chart with regime bands ──────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(prices.loc[idx, primary], color='#2c3e50', linewidth=0.8, label='Nifty 50')
    for regime, colour in REGIME_COLORS.items():
        mask = regime_series.loc[idx] == regime
        if mask.any():
            ax.fill_between(idx, prices.loc[idx, primary].min(),
                            prices.loc[idx, primary].max(),
                            where=mask, alpha=0.18, color=colour, label=regime)
    ax.set_title('Nifty 50 with HMM Regime Overlay', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_ylabel('Price')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'regime_overlay.png'), dpi=150)
    print("  Saved regime_overlay.png")

    # ── 2. Equity curves ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    colors_eq = ['#2ecc71', '#27ae60', '#3498db', '#e67e22']
    for (name, eq), c in zip(equities.items(), colors_eq):
        ax.plot(eq, label=name, linewidth=1.2, color=c)
    ax.set_title('Equity Curves — Dynamic vs Static Benchmarks', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylabel('Growth of $1')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'equity_curves.png'), dpi=150)
    print("  Saved equity_curves.png")

    # ── 3. Portfolio weight evolution ─────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    w = weights_df.loc[idx]
    ax.stackplot(w.index, *[w[c] for c in w.columns],
                 labels=[ASSETS.get(c, c) for c in w.columns],
                 alpha=0.85, colors=['#3498db', '#f1c40f', '#1abc9c'])
    ax.set_title('Dynamic Portfolio Weight Allocation Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'weight_evolution.png'), dpi=150)
    print("  Saved weight_evolution.png")

    # ── 4. Transition matrix of the full-sample HMM ──────────
    # (for the submission — fit once on full data for illustration)
    # This is separate from the walk-forward logic.
    print("  Plots saved to output/ directory.")
    plt.close('all')


def fit_full_sample_hmm(features, returns, primary='^NSEI'):
    """Fit HMM on full dataset for visualisation of transition matrix."""
    print("\n  Fitting full-sample HMM for transition matrix display...")
    common = features.index.intersection(returns.index)
    feat = features.loc[common]
    ret  = returns.loc[common]

    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)
    model = fit_hmm(X)
    if model is None:
        print("  WARNING: Full-sample HMM failed to converge.")
        return

    labels, smap = map_states_to_regimes(model, X, ret, primary)
    inv_map = {v: k for k, v in smap.items()}
    order = [inv_map.get('Bull', 0), inv_map.get('Bear', 1), inv_map.get('Crisis', 2)]

    trans = model.transmat_[np.ix_(order, order)]

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │       Regime Transition Matrix           │")
    print("  ├─────────┬──────────┬──────────┬──────────┤")
    print("  │         │   Bull   │   Bear   │  Crisis  │")
    print("  ├─────────┼──────────┼──────────┼──────────┤")
    for i, name in enumerate(['Bull', 'Bear', 'Crisis']):
        row = "  │ {:7s} │".format(name)
        for j in range(3):
            row += "  {:.4f}  │".format(trans[i, j])
        print(row)
    print("  └─────────┴──────────┴──────────┴──────────┘")

    # Save transition matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(trans, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{trans[i,j]:.3f}", ha='center', va='center', fontsize=12)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Bull', 'Bear', 'Crisis'])
    ax.set_yticklabels(['Bull', 'Bear', 'Crisis'])
    ax.set_title('HMM Transition Probability Matrix', fontsize=13, fontweight='bold')
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'transition_matrix.png'), dpi=150)
    print("  Saved transition_matrix.png")
    plt.close()

    return model, labels


# ═════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  HMM Regime Detection & Dynamic Portfolio Optimisation  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Phase 1
    prices, returns, vix = fetch_data()

    # Phase 2
    features = compute_features(returns, vix)

    # Phase 3 + 4: Walk-forward (trains HMM inside)
    regime_series, weights_df, rebal_dates = walk_forward(features, returns)

    # Phase 6: Backtest
    strats, equities, metrics_df = backtest(returns, weights_df, rebal_dates)

    # Phase 7: Visualisation
    plot_all(prices, regime_series, equities, weights_df, metrics_df)

    # Full-sample HMM for transition matrix (submission artefact)
    fit_full_sample_hmm(features, returns)

    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_summary.csv'), index=False)
    print(f"\n  Metrics saved to {OUTPUT_DIR}/performance_summary.csv")

    print("\n" + "═" * 60)
    print(" ✓  Pipeline complete. Check the output/ folder.")
    print("═" * 60)


if __name__ == '__main__':
    main()
