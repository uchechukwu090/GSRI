# scripts/run_gsri_and_commit.py
"""
Run GSRI pipeline, write a small snapshot (latest row) to gsri_snapshot.json.
Designed to be run inside GitHub Actions. It does NOT perform git operations itself.
"""

import json
from pathlib import Path
import numpy as np

# Import your GSRI module. Adjust import path if you place GSRI.py in a package.
from GSRI import compute_gsri, download_price_data, fetch_sp500_tickers, START_DATE, END_DATE, WINDOW

OUT_PATH = Path("gsri_snapshot.json")

def main():
    tickers = fetch_sp500_tickers()
    data = download_price_data(tickers, START_DATE, END_DATE)
    data = data.ffill().bfill()
    returns = np.log(data / data.shift(1)).dropna(how="all")

    # If returns are too short, write a conservative snapshot and exit
    if returns.shape[0] < WINDOW * 2:
        snapshot = [{"Risk_Score": 0.8, "Alert": 1}]
        OUT_PATH.write_text(json.dumps(snapshot, indent=2))
        print("Not enough history; wrote conservative fallback snapshot.")
        return

    gsri_df = compute_gsri(returns, window=WINDOW)

    # Keep only the last row to keep file small
    latest = gsri_df.tail(1).to_dict(orient="records")
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    latest_clean = [{k: convert(v) for k, v in row.items()} for row in latest]
    OUT_PATH.write_text(json.dumps(latest_clean, indent=2))
    print(f"Wrote snapshot to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
