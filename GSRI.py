#!/usr/bin/env python3
"""
GSRI.py - Bulletproof Real Data Pipeline
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.stats import entropy

# ---------------------------
# Configuration
# ---------------------------
SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
# Filtered fallback: Stocks guaranteed to have 15+ years of clean data
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
    "JNJ", "XOM", "WMT", "PG", "BAC",
    "HD", "CVX", "PFE", "KO", "MRK",
    "PEP", "CSCO", "INTC", "VZ", "C"
]
MAX_TICKERS = 100
START_DATE = "2006-01-01"
END_DATE = "2021-12-31"
WINDOW = 40
MIN_COLS_REQUIRED = 5 


# ---------------------------
# GSRI Engine (Your Final Math)
# ---------------------------
def compute_gsri(returns_df: pd.DataFrame, window: int = 40) -> pd.DataFrame:
    if returns_df is None or returns_df.shape[1] == 0:
        raise ValueError("compute_gsri requires returns_df with at least one column")

    N = returns_df.shape[1]
    length = len(returns_df)

    metrics = {
        "Norm_Entropy": np.full(length, np.nan),
        "Risk_Score": np.full(length, np.nan),
        "Alert": np.zeros(length, dtype=int),
    }

    tau_history, conc_history, score_history = [], [], []
    K_prev = None

    for t in range(window, length):
        R_t = returns_df.iloc[t - window : t].values 
        if R_t.shape[1] == 0: continue

        col_mean = R_t.mean(axis=0)
        col_std = R_t.std(axis=0)
        col_std[col_std < 1e-6] = 1e-6
        R_std = (R_t - col_mean) / col_std

        try:
            lw = LedoitWolf().fit(R_std)
            cov = lw.covariance_
            G_t = lw.precision_
        except Exception:
            continue

        try:
            K_t = np.trace(G_t) / float(N)
        except Exception:
            K_t = 0.0

        tau_t = ((K_t - K_prev) / (abs(K_prev) + 1e-8)) if K_prev is not None else 0.0
        K_prev = K_t

        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eig_sum = np.sum(eigenvalues) + 1e-12
        lambda_1 = eigenvalues[0] if eigenvalues.size > 0 else 0.0

        concentration = float(lambda_1 / eig_sum) if eig_sum != 0 else 0.0

        p = eigenvalues / eig_sum if eig_sum != 0 else np.ones_like(eigenvalues) / max(1, len(eigenvalues))
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        norm_ent = float(entropy(p) / np.log(len(p))) if len(p) > 1 else 0.0

        tau_history.append(tau_t)
        conc_history.append(concentration)
        metrics["Norm_Entropy"][t] = norm_ent

        if len(tau_history) > (window * 2):
            baseline_slice = slice(0, -window)
            tau_baseline = np.array(tau_history[baseline_slice])

            tau_mean = tau_baseline.mean()
            tau_std = tau_baseline.std()

            tau_z = np.clip((tau_t - tau_mean) / (tau_std + 1e-8), 0, None)
            tau_norm = np.tanh(tau_z)

            ent_norm = 1.0 - norm_ent
            conc_norm = np.clip(concentration, 0.0, 1.0)

            risk_score = float(np.mean([tau_norm, ent_norm, conc_norm]))
            score_history.append(risk_score)
            metrics["Risk_Score"][t] = risk_score

            if len(score_history) > (window * 2):
                score_baseline = np.array(score_history[baseline_slice])
                if len(score_baseline) < 30: continue

                thresh_mean = np.mean(score_baseline) + 2.0 * np.std(score_baseline)
                thresh_pct = np.percentile(score_baseline, 90)
                score_thresh = max(thresh_mean, thresh_pct)

                if len(score_history) >= 2 and all(s > score_thresh for s in score_history[-2:]):
                    metrics["Alert"][t] = 1

    return pd.DataFrame(metrics, index=returns_df.index)


# ---------------------------
# Utilities
# ---------------------------
def fetch_sp500_tickers(max_tickers: int = MAX_TICKERS) -> list:
    try:
        resp = requests.get(SP500_WIKI, headers=HEADERS, timeout=10)
        if resp.status_code != 200: return FALLBACK_TICKERS
        tables = pd.read_html(resp.text)
        if not tables: return FALLBACK_TICKERS
        df = tables[0]
        
        col = "Symbol"
        if col not in df.columns:
            possible = [c for c in df.columns if "Symbol" in str(c) or "Ticker" in str(c)]
            col = possible[0] if possible else None
            
        if not col: return FALLBACK_TICKERS
        
        symbols = df[col].astype(str).tolist()
        symbols = [s.replace(".", "-").strip() for s in symbols if isinstance(s, str) and s.strip()]
        return symbols[:max_tickers] if len(symbols) >= 1 else FALLBACK_TICKERS
    except Exception:
        return FALLBACK_TICKERS


def download_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    if not tickers: raise ValueError("No tickers")
    print(f"[download] Fetching {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end, progress=False, threads=True)

    data = None
    # Modern yfinance returns chaotic MultiIndex formats. This handles all of them.
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0, lvl1 = raw.columns.get_level_values(0).unique(), raw.columns.get_level_values(1).unique()
        target = None
        if 'Adj Close' in lvl0 or 'Adj Close' in lvl1: target = 'Adj Close'
        elif 'Close' in lvl0 or 'Close' in lvl1: target = 'Close'
        
        if target:
            for lvl in [0, 1]:
                try:
                    extracted = raw.xs(target, axis=1, level=lvl)
                    if not extracted.empty:
                        data = extracted
                        break
                except KeyError:
                    continue
                    
        if data is None or data.empty:
            if raw.columns.get_level_values(0).isin(['Close', 'Adj Close']).any():
                data = raw.copy()
                data.columns = raw.columns.droplevel(0)
            else:
                data = raw.copy()
                data.columns = [f"{c[0]}_{c[1]}" if len(c)==2 else c[0] for c in raw.columns]
    else:
        data = raw

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data


# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    start_time = time.time()

    tickers = fetch_sp500_tickers(MAX_TICKERS)
    data = download_price_data(tickers, START_DATE, END_DATE)
    print(f"[main] Raw data shape: {data.shape}")

    # FIX 1: Lower threshold to 50%. 90% wiped out valid recent IPOs.
    thresh = int(0.5 * len(data))
    data = data.dropna(axis=1, thresh=thresh)
    
    # FIX 2: Modern pandas syntax
    data = data.ffill().bfill()

    # FIX 3: True Fallback Mechanism
    if data.shape[1] < MIN_COLS_REQUIRED:
        print(f"[main] Warning: Only {data.shape[1]} tickers survived. Using legacy fallback list...")
        data = download_price_data(FALLBACK_TICKERS, START_DATE, END_DATE)
        data = data.dropna(axis=1, thresh=thresh).ffill().bfill()

    if data.shape[1] < MIN_COLS_REQUIRED:
        raise RuntimeError(f"FATAL: Not enough tickers: {data.shape[1]} (need >= {MIN_COLS_REQUIRED})")

    print(f"[main] Cleaned data shape: {data.shape}")

    # Compute returns
    returns = np.log(data / data.shift(1)).dropna(how="all")
    col_std = returns.std(axis=0)
    returns = returns.clip(lower=-4.0 * col_std, upper=4.0 * col_std, axis=1)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    
    if returns.shape[0] < WINDOW * 3:
        raise RuntimeError("Not enough time series length.")

    print("[main] Running GSRI engine...")
    gsri_results = compute_gsri(returns, window=WINDOW)

    # Event compression
    alert_indices = np.where(gsri_results["Alert"] == 1)[0]
    compressed_alerts = []
    last_alert = -np.inf
    cooldown = int(WINDOW * 0.8)
    for idx in alert_indices:
        if idx - last_alert > cooldown:
            compressed_alerts.append(idx)
            last_alert = idx

    crisis_periods = [
        ("2007-10-01", "2008-10-01"), ("2010-04-01", "2010-07-01"),
        ("2011-07-01", "2011-10-01"), ("2015-08-01", "2015-09-15"),
        ("2018-02-01", "2018-04-01"), ("2020-02-15", "2020-04-15"),
    ]

    dates = returns.index
    crisis_zones = []
    for start, end in crisis_periods:
        s_idx = np.searchsorted(dates, pd.to_datetime(start))
        e_idx = np.searchsorted(dates, pd.to_datetime(end))
        crisis_zones.append((s_idx, e_idx))

    # Validation
    detected, false_pos, lead_times = set(), 0, []
    for idx in compressed_alerts:
        hit = False
        for i, (c_start, c_end) in enumerate(crisis_zones):
            if (c_start - 10) <= idx <= c_end:
                detected.add(i); hit = True
                if idx <= c_start: lead_times.append(c_start - idx)
                break
        if not hit: false_pos += 1

    precision = len(detected) / len(compressed_alerts) if compressed_alerts else 0.0
    recall = len(detected) / len(crisis_zones) if crisis_zones else 0.0
    fpr = false_pos / len(returns) * 100.0
    lead = float(np.mean(lead_times)) if lead_times else 0.0

    print("\n" + "=" * 50)
    print("REAL DATA VALIDATION (HARSH TRUTH)")
    print("=" * 50)
    print(f"Target Crises: {len(crisis_zones)}")
    print(f"Detected: {len(detected)} (Recall: {recall:.2f})")
    print(f"Total Alert Blocks: {len(compressed_alerts)}")
    print(f"Event Precision: {precision:.2f}")
    print(f"False Positives: {false_pos} ({fpr:.2f} per 100 days)")
    if lead_times:
        print(f"True Early Warnings: {len(lead_times)}")
        print(f"Avg Lead Time: {lead:.1f} days")
    else:
        print("Avg Lead Time: N/A")
    print("=" * 50 + "\n")

    # Visualization
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        spy = download_price_data(["SPY"], START_DATE, END_DATE)
        
        spy_col = [c for c in spy.columns if 'SPY' in str(c)][0]
        axes[0].plot(spy.index, spy[spy_col].values, label="SPY", color="blue")
        
        for s_idx, e_idx in crisis_zones:
            if 0 <= s_idx < len(dates) and 0 <= e_idx <= len(dates):
                axes[0].axvspan(dates[s_idx], dates[min(e_idx, len(dates)-1)], alpha=0.2, color="red")
        axes[0].set_title("Market (SPY) vs. Known Systemic Stress Zones"); axes[0].legend()

        axes[1].plot(gsri_results.index, gsri_results["Norm_Entropy"], color="purple")
        axes[1].set_ylabel("Entropy (0-1)"); axes[1].set_title("Dimensional Freedom (Drops = Systemic Lockstep)")

        axes[2].plot(gsri_results.index, gsri_results["Risk_Score"], color="red", linewidth=1)
        axes[2].fill_between(gsri_results.index, 0, gsri_results["Alert"], step="mid", alpha=0.5, color="black")
        for idx in compressed_alerts:
            if 0 <= idx < len(gsri_results.index):
                axes[2].axvline(gsri_results.index[idx], linestyle="--", color="lime", alpha=0.7)
        axes[2].set_title("GSRI Risk Score + Compressed Alerts")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[main] Visualization failed: {e}")

    print(f"[main] Done. Elapsed: {time.time() - start_time:.1f}s.")

if __name__ == "__main__":
    main()