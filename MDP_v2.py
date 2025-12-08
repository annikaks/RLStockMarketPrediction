import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import random

plt.rcParams["figure.figsize"] = (10, 5)
#fixed random seed to get same results
random.seed(2)
np.random.seed(2)


DATA_DIR = "nasdaq_data/"        # folder containing all ticker CSVs
TICKER   = "NVDA"                # choose a ticker inside dataset
K_WINDOW = 10                    # k: number of past returns in state
N_EPISODES = 200                 # Q-learning episodes
ALPHA = 0.1                      # learning rate
GAMMA = 0.99                     # discount
EPSILON = 0.1                    # exploration

# date split
TRAIN_START = "2015-01-01"
TRAIN_END   = "2024-06-01"
TEST_START  = "2024-06-10"
TEST_END    = "2025-12-03"


#populating dataframe
all_dfs = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".csv"):
        path = os.path.join(DATA_DIR, filename)
        df_temp = pd.read_csv(path)

        ticker = filename.split(".")[0].upper()
        df_temp["Name"] = ticker

        all_dfs.append(df_temp)

df = pd.concat(all_dfs).reset_index(drop=True)

print("Loaded tickers:", df["Name"].unique())
print("Total rows:", len(df))


#split data based on dates
df_ticker = df[df["Name"] == TICKER].copy().reset_index(drop=True)

if len(df_ticker) == 0:
    raise ValueError(f"No rows found for ticker {TICKER}")

df_ticker["date"] = pd.to_datetime(df_ticker["date"])
df_ticker = df_ticker.sort_values("date").reset_index(drop=True)

train_df = df_ticker[(df_ticker["date"] >= TRAIN_START) & (df_ticker["date"] <= TRAIN_END)]
test_df  = df_ticker[(df_ticker["date"] >= TEST_START)  & (df_ticker["date"] <= TEST_END)]

train_prices = train_df["close"].astype(float).values
test_prices  = test_df["close"].astype(float).values

print("Train window size:", len(train_prices))
print("Test window size:", len(test_prices))

# check that our window fits into dataset size
if len(test_prices) < K_WINDOW + 2:
    raise ValueError("Test window too small for chosen K_WINDOW")


#mdp definition
class TradingMDP:
    def __init__(self, prices, k=10, initial_pos=0):
        self.prices = np.array(prices, dtype=float)
        self.k = k
        self.initial_pos = initial_pos


        self.returns = (self.prices[1:] - self.prices[:-1]) / self.prices[:-1]

        self.reset()

    def reset(self):
        self.t = self.k
        self.pos = self.initial_pos  # 0 = no position, 1 = long one share
        return self._get_state()

    def _get_state(self):
        window = self.returns[self.t - self.k : self.t]
        return np.concatenate([window, [self.pos]])

    def step(self, action):
        assert action in [-1, 1]

        p_t = self.prices[self.t]
        p_tp1 = self.prices[self.t + 1]

        # update position
        self.pos = 1 if action == 1 else 0

        # reward
        reward = (p_tp1 - p_t) if self.pos == 1 else 0.0

        # step forward
        self.t += 1
        done = (self.t >= len(self.prices) - 1)

        next_state = None if done else self._get_state()
        return next_state, reward, done

train_env = TradingMDP(train_prices, k=K_WINDOW)
test_env  = TradingMDP(test_prices, k=K_WINDOW)


#create discretized buckets
def discretize_state(state, ret_threshold=0.005):
    if state is None:
        return None

    k = len(state) - 1
    pos = int(state[-1])
    buckets = []

    for r in state[:k]:
        if r > ret_threshold:
            buckets.append(1)
        elif r < -ret_threshold:
            buckets.append(-1)
        else:
            buckets.append(0)

    return tuple(buckets + [pos])

# check our discretization is working
print("Example discrete:", discretize_state(train_env.reset()))


#q learning algo
ACTIONS = [1, -1]
Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

def epsilon_greedy_action(d_state, epsilon=EPSILON):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return ACTIONS[int(np.argmax(Q[d_state]))]


def q_learning(env, n_episodes, alpha, gamma, epsilon):
    global Q
    ep_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        d_state = discretize_state(state)
        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy_action(d_state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                Q[d_state][ACTIONS.index(action)] += alpha * (reward - Q[d_state][ACTIONS.index(action)])
                break

            d_next_state = discretize_state(next_state)
            q_sa = Q[d_state][ACTIONS.index(action)]
            q_next = np.max(Q[d_next_state])
            target = reward + gamma * q_next

            Q[d_state][ACTIONS.index(action)] = q_sa + alpha * (target - q_sa)
            d_state = d_next_state

        ep_rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}: reward {total_reward:.2f}")

    return ep_rewards


episode_rewards = q_learning(train_env, N_EPISODES, ALPHA, GAMMA, EPSILON)


#plot for training curve
# plt.plot(episode_rewards)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title(f"Training Rewards for Q-learning ({TICKER})")
# plt.grid(True)
# plt.show()


#see how much profit we make
def run_backtest(env, initial_cash=10000):
    state = env.reset()
    d_state = discretize_state(state)
    done = False

    cash = initial_cash
    shares = 0
    prices = env.prices
    portfolio_values = []

    while not done:
        # greedy policy (no exploration)
        q_vals = Q[d_state]
        action = ACTIONS[int(np.argmax(q_vals))]

        price = prices[env.t]

        if action == 1 and shares == 0:    # BUY
            shares = cash / price
            cash = 0
        elif action == -1 and shares > 0:  # SELL
            cash = shares * price
            shares = 0


        next_state, reward, done = env.step(action)
        if not done:
            d_state = discretize_state(next_state)

        # curr portfolio = cash + shares * price
        current_value = cash + shares * prices[env.t]
        portfolio_values.append(current_value)


    final_value = portfolio_values[-1]
    return final_value, portfolio_values


final_value, value_curve = run_backtest(test_env, initial_cash=100)

print("========================================")
print(f"Final Portfolio Value: ${final_value:,.2f}")
print(f"Total Profit:     ${final_value - 100:,.2f}")
print(f"Percent Return:        {(final_value - 100)/100 * 100:.2f}%")
print("========================================")

#see how portfolio value changes over time
# plt.plot(value_curve)
# plt.title("RL Agent Portfolio Value Over Test Period")
# plt.xlabel("Days")
# plt.ylabel("Portfolio Value ($)")
# plt.grid(True)
# plt.show()


# calculating our q-learning test rewards without considering intial cash
def q_rewards(env):
    state = env.reset()
    d_state = discretize_state(state)
    done = False

    holding = 1 if state[-1] == 1 else 0

    prices = env.prices
    cumulative_profit = 0.0

    buy_count = 0
    sell_count = 0

    while not done:

        q_vals = Q[d_state]
        action = ACTIONS[int(np.argmax(q_vals))]


        if action == 1:
            buy_count += 1
        elif action == -1:
            sell_count += 1


        if action == 1:
            holding = 1
        elif action == -1:
            holding = 0


        if holding == 1:
            p_t = prices[env.t]
            p_tp1 = prices[env.t + 1]
            cumulative_profit += (p_tp1 - p_t)


        next_state, reward, done = env.step(action)
        if not done:
            d_state = discretize_state(next_state)

    return cumulative_profit, buy_count, sell_count


def buy_and_hold_baseline(prices):
    return float(prices[-1] - prices[0])



q_learning_reward, buys, sells = q_rewards(test_env)
baseline_reward = buy_and_hold_baseline(test_env.prices)

print("Buy-hold baseline:", baseline_reward)
print("Q-learning reward:", q_learning_reward)
print("Number of BUY actions:", buys)
print("Number of SELL actions:", sells)



