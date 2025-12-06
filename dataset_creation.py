import yfinance as yf
import pandas as pd
import csv
import os
from config import TICKERS, START_DATE, END_DATE

saved_files = []

for ticker_symbol in TICKERS:
    # ticker_data = yf.Ticker(ticker_symbol)
    try:
        data = yf.download(ticker_symbol, start=START_DATE, end=END_DATE)
        if not data.empty and 'Close' in data.columns:
            out_path = f"data/{ticker_symbol}_daily.csv"
            if os.path.exists(out_path):
                os.remove(out_path)
            
            data_to_write = data.reset_index()[['Date', 'Close']]
            header = ['Date', 'Cost']
            with open(out_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header) 
                writer.writerows(data_to_write.values)


            print(f"saving {ticker_symbol} data to {out_path}: {len(data)} data loaded")
            saved_files.append(out_path)
        
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