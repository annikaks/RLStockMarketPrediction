import yfinance as yf
import pandas as pd
from config import TICKERS, START_DATE, END_DATE

saved_files = []

for ticker_symbol in TICKERS:
    ticker_data = yf.Ticker(ticker_symbol)
    try:
        history = ticker_data.history(period="1d", raise_errors=True)
        if not history.empty:
            data = yf.download(ticker_symbol, start=START_DATE, end=END_DATE)

            data = data[["Close"]]
            out_path = f"data/{ticker_symbol}_daily.csv"
            data.to_csv(out_path, index_label="Date")

            print(f"saving {ticker_symbol} data to {out_path}: {len(data)} data loaded")
            saved_files.append(out_path)
        else:
            print(f"{ticker_symbol}: no history")
    except Exception as e:
        print(f"{ticker_symbol} not available")
        print(f"    {e}")

# write data path files to dynamic .py file (saves file names)
with open("valid_data_paths.py", "w") as f:
    f.write("# AUTO-GENERATED. DO NOT EDIT.\n")
    f.write("DATA_FILES = [\n")
    for fp in saved_files:
        f.write(f"    '{fp}',\n")
    f.write("]\n")