import yfinance as yf
import pandas as pd
import csv
import os
from config import TICKERS, START_DATE, END_DATE, LOGGING, DATASET_CREATION_LOGGING, TRAIN_TEST_SPLIT_DATE

saved_train_files = []
saved_test_files = []
log = LOGGING or DATASET_CREATION_LOGGING

def parse_and_save_data():
    for ticker_symbol in TICKERS:
        # ticker_data = yf.Ticker(ticker_symbol)
        try:
            data = yf.download(ticker_symbol, start=START_DATE, end=END_DATE)

            if not data.empty and 'Close' in data.columns:
                df = data.reset_index()[['Date', 'Close']]
                out_path = f"data/{ticker_symbol}_daily.csv"

                train_df = df[df['Date'] < TRAIN_TEST_SPLIT_DATE]
                test_df  = df[df['Date'] >= TRAIN_TEST_SPLIT_DATE]
                train_path = f"data/{ticker_symbol}_train.csv"
                test_path  = f"data/{ticker_symbol}_test.csv"
                
                if os.path.exists(out_path): os.remove(out_path)
                if os.path.exists(train_path): os.remove(train_path)
                if os.path.exists(test_path): os.remove(test_path)
                
                # data_to_write = data.reset_index()[['Date', 'Close']]
                header = ['Date', 'Cost']
                with open(train_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header) 
                    writer.writerows(train_df.values)

                with open(test_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header) 
                    writer.writerows(test_df.values)

                if log: print(f"        saving {ticker_symbol} data to {train_path}: {len(train_df)} data loaded")
                if log: print(f"        saving {ticker_symbol} data to {test_path}: {len(test_df)} data loaded")
                saved_train_files.append(train_path)
                saved_test_files.append(test_path)

        except Exception as e:
            if log: print(f"        {ticker_symbol} not available")
            if log: print(f"            {e}")

    # write data path files to dynamic .py file (saves file names)
    with open("valid_data_paths.py", "w") as f:
        f.write("# AUTO-GENERATED. DO NOT EDIT.\n")
        f.write("TRAIN_FILES = [\n")
        for fp in saved_train_files:
            f.write(f"    '{fp}',\n")
        f.write("]\n")
    
    with open("valid_test_paths.py", "w") as f:
        f.write("# AUTO-GENERATED. DO NOT EDIT.\n")
        f.write("TEST_FILES = [\n")
        for fp in saved_test_files:
            f.write(f"    '{fp}',\n")
        f.write("]\n")

def main():
    if log: print(f"dataset_creation.py:")
    parse_and_save_data()
    if log: print("!!! COMPLETED\n")

if __name__ == "__main__":
    main()
    