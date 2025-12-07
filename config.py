from valid_data_paths import DATA_FILES

# DATA configs
# dataset_creation.py, dataset_preparation.py
TICKERS = ["NDAQ"]
START_DATE = "2020-01-01"
END_DATE = "2025-12-03"
TRAIN_TEST_SPLIT_DATE = "2025-01-01"

DATA_FILES = DATA_FILES
NUM_DISCRETE_CATEGORIES = 5

# MDP SETUP
# trading_mdp.py
ACTIONS = [-1, 0, 1]
HISTORY_WINDOW_LENGTH = 5

# QLEARNING SETUP
ALPHA = 0.1 # lr
GAMMA = 0.99 # discout
EPSILON = 0.5 # explor



# LOGGING config
# note that setting logging to true will OVERRIDE all other logging configs
LOGGING = True # activate all logging
DATASET_CREATION_LOGGING = False
DATASET_PREPARATION_LOGGING = True
Q_LEARNING_AGENT_LOGGING = True