# index.py
"""
Minimal trading entry that consumes a GSRI snapshot and applies
a simple intraday-friendly risk overlay (soft sizing + optional hedge).
Designed to be safe by default and easy to extend.
"""

import os
import json
import time
import math
import requests
from pathlib import Path

# CONFIG
# If you commit gsri_snapshot.json to the repo, set LOCAL_SNAPSHOT=True.
LOCAL_SNAPSHOT = True
LOCAL_SNAPSHOT_PATH = Path("gsri_snapshot.json")

# If you prefer to fetch from the raw GitHub URL, set this to False and provide RAW_URL.
RAW_URL = "https://raw.githubusercontent.com/yourusername/yourrepo/main/gsri_snapshot.json"
REMOTE_TIMEOUT = 8  # seconds

# Risk-to-sizing mapping (tune these)
BASE_POSITION_SIZE = 1.0        # nominal size for a trade
MIN_POSITION_SCALE = 0.2        # never go below this fraction of base size
ALPHA = 0.6                     # sensitivity to Risk_Score for soft sizing

# Hedge settings (simple example)
ENABLE_HEDGE = True
HEDGE_ON_ALERT = True           # if True, open/maintain hedge when Alert==1
HEDGE_SIZE_FRAC = 0.15          # fraction of portfolio to hedge when alert

# Fallback conservative defaults if snapshot unavailable
FALLBACK_RISK_SCORE = 0.8
FALLBACK_ALERT = 1

def load_local_snapshot(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if not data:
            return None
        return data[-1]
    except Exception:
        return None

def fetch_remote_snapshot(raw_url: str):
    try:
        r = requests.get(raw_url, timeout=REMOTE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return data[-1]
    except Exception:
        return None

def get_latest_gsri():
    # Priority: local file (fast) -> remote raw URL -> fallback
    if LOCAL_SNAPSHOT:
        snap = load_local_snapshot(LOCAL_SNAPSHOT_PATH)
        if snap:
            return snap
    snap = fetch_remote_snapshot(RAW_URL)
    if snap:
        return snap
    # fallback
    return {"Risk_Score": FALLBACK_RISK_SCORE, "Alert": FALLBACK_ALERT}

def compute_position_scale(risk_score: float, alpha: float = ALPHA) -> float:
    """Soft sizing: scale in [MIN_POSITION_SCALE, 1.0]"""
    scale = max(MIN_POSITION_SCALE, 1.0 - alpha * float(risk_score))
    return float(scale)

def decide_trade(signal_strength: float, risk_scale: float, allow_new_entries: bool):
    """
    Example decision function.
    - signal_strength: your alpha signal in [-1,1] (positive = long)
    - risk_scale: from compute_position_scale
    - allow_new_entries: bool
    Returns target position size (signed).
    """
    if not allow_new_entries:
        return 0.0
    size = BASE_POSITION_SIZE * risk_scale * abs(signal_strength)
    return math.copysign(size, signal_strength)

def maybe_open_hedge(alert_flag: int):
    """Simple hedge decision: returns hedge fraction to apply."""
    if not ENABLE_HEDGE:
        return 0.0
    if HEDGE_ON_ALERT and alert_flag:
        return HEDGE_SIZE_FRAC
    return 0.0

# -------------------------
# Example usage in a loop
# -------------------------
if __name__ == "__main__":
    # Example: replace this with your real alpha signal source
    example_alpha_signal = 0.7  # long bias (intraday signal magnitude)

    latest = get_latest_gsri()
    risk_score = float(latest.get("Risk_Score", FALLBACK_RISK_SCORE))
    alert = int(latest.get("Alert", FALLBACK_ALERT))

    # For intraday: you may want to be more reactive; for swing: smoother.
    # This snippet is neutral: it reads the latest snapshot and applies soft sizing.
    risk_scale = compute_position_scale(risk_score)
    allow_new_entries = (alert == 0)  # simple rule: block new entries on alert

    target_size = decide_trade(example_alpha_signal, risk_scale, allow_new_entries)
    hedge_frac = maybe_open_hedge(alert)

    print("GSRI latest:", latest)
    print(f"Computed risk_scale: {risk_scale:.3f}")
    print(f"Allow new entries: {allow_new_entries}")
    print(f"Target position size (signed): {target_size:.4f}")
    print(f"Hedge fraction to apply: {hedge_frac:.3f}")

    # Integrate with your execution system here:
    # - If target_size != current_size: send order to adjust
    # - If hedge_frac > 0: open/maintain hedge instrument (e.g., inverse ETF or options)
