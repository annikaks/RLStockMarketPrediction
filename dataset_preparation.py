import pandas as pd
import numpy as np
import json
from config import DATA_FILES, LOGGING, DATASET_PREPARATION_LOGGING, NUM_DISCRETE_CATEGORIES

log = LOGGING or DATASET_PREPARATION_LOGGING
# return array of prices from csv
def load_daily_data(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df = df.sort_values("Date")
    prices = df["Close"].astype(float).reset_index(drop=True) 
    return prices

# compute daily percentage change
def compute_returns(prices: pd.Series) -> np.ndarray:
    p = prices.values
    r = np.zeros_like(p, dtype=float)
    r[1:] = (p[1:] - p[:-1]) / p[:-1] # percent change from yday
    return r

# sort cost shifts into n_bins of equal size
def discretize_returns_qcut(returns: np.ndarray, n_bins: int = 5):
    s = pd.Series(returns)

    cats, bins = pd.qcut(s, q=n_bins, labels=False, retbins=True, duplicates="drop")
    return cats.to_numpy(), bins

def main():
    if log: print(f"dataset_preparation.py:")
    discretization_map = {}

    for csv_path in DATA_FILES:
        prices = load_daily_data(csv_path)
        price_returns = compute_returns(prices)

        cats, bins = discretize_returns_qcut(price_returns, n_bins=NUM_DISCRETE_CATEGORIES) # sort into n_bins. 0 -> very bad day ... n_bin - 1 -> very good day 
        if log: print(f"     Processing {csv_path}")
        if log: print("     Bin edges:", bins)
        if log: print(f"     Category counts:{np.bincount(cats)}\n")
        discretization_map[csv_path] = {
            "cats": cats.tolist(),
            "bins": bins.tolist(),
        }

    out_path = "data/discretized_returns.json"
    with open(out_path, "w") as f:
        json.dump(discretization_map, f, indent=2)
    if log: print("!!! COMPLETED\n")

if __name__ == "__main__":
    main()