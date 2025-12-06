# DATA configs
# dataset_creation.py, dataset_preparation.py
from valid_data_paths import DATA_FILES
TICKERS = ["NDAQ", "VOO"]
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

DATA_FILES = DATA_FILES


# LOGGING config
# note that setting logging to true will OVERRIDE all other logging configs
LOGGING = True # activate all logging
DATASET_CREATION_LOGGING = False
DATASET_PREPARATION_LOGGING = True