import pandas as pd
import numpy as np
from config import DATA_FILES, LOGGING, DATASET_PREPARATION_LOGGING, NUM_DISCRETE_CATEGORIES

log = LOGGING or DATASET_PREPARATION_LOGGING
# return array of prices from csv
def load_daily_data(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df = df.sort_values("Date")
    prices = df["Cost"].astype(float).reset_index(drop=True) 
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

if __name__ == "__main__":
    if log: print(f"dataset_preparation.py:")
    for csv_path in DATA_FILES:
        prices = load_daily_data(csv_path)
        price_returns = compute_returns(prices)

        cats, bins = discretize_returns_qcut(price_returns, n_bins=NUM_DISCRETE_CATEGORIES) # sort into n_bins. 0 -> very bad day ... n_bin - 1 -> very good day 
        if log: print(f"     Processing {csv_path}")
        if log: print("     Bin edges:", bins)
        if log: print(f"     Category counts:{np.bincount(cats)}\n")
    if log: print("!!! COMPLETED")
