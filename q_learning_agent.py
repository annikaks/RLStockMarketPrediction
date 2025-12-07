import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict
from collections import defaultdict
import random
import os
import json

from config import HISTORY_WINDOW_LENGTH, NUM_DISCRETE_CATEGORIES, ACTIONS, ALPHA, GAMMA, EPSILON, DATA_FILES, Q_LEARNING_AGENT_LOGGING, LOGGING, TRAIN_TEST_SPLIT_DATE, NUM_EPISODES, EPSILON_DECAY
from trading_mdp import TradingState, TradingMDP
import dataset_creation, dataset_preparation

class QLearningAgent:
    def __init__(self, mdp: TradingMDP, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.5):
        self.mdp = mdp
        self.alpha = ALPHA     
        self.gamma = GAMMA     
        self.epsilon = EPSILON 

        self.Q: Dict[Tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(ACTIONS))) # <encoded state tuple, qval of actions [-1, 0, 1]>
        self.actions = self.mdp.actions

        self.action_map = {action: i for i, action in enumerate(self.actions)} # turn actions (-1, 0, 1) to indecies -> (0, 1, 2)

    def get_q_values(self, s_encoded: Tuple[int, ...]) -> np.ndarray:
        return self.Q[s_encoded]

    def get_best_action(self, s_encoded: Tuple[int, ...]) -> int:
        q_values = self.get_q_values(s_encoded)
        best_action_index = np.argmax(q_values)
        return self.actions[best_action_index]

    def choose_action(self, s_encoded: Tuple[int, ...]) -> int:
        # using epsilon-greedy policy for action (explore v exploit)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.get_best_action(s_encoded)
            
    def train(self, num_episodes: int, epsilon_decay: float = 0.999) -> list[float]:
        total_rewards = []
        
        for episode in range(num_episodes):
            s = self.mdp.initial_state()
            s_encoded = self.mdp.encode_state(s)
            
            episode_reward = 0
            done = False
            
            while not done:
                a = self.choose_action(s_encoded)
                a_index = self.action_map[a]
                
                s_next, reward, done = self.mdp.step(s, a)
                s_next_encoded = self.mdp.encode_state(s_next)
                episode_reward += reward

                # bellman eq
                if not done:
                    Q_sa = self.get_q_values(s_encoded)[a_index]
                    max_q_next = np.max(self.get_q_values(s_next_encoded))
                    
                    # Q(s,a) <- Q(s,a) + α [R + γ * max Q(s',a') - Q(s,a)]
                    new_q_value = Q_sa + self.alpha * (reward + self.gamma * max_q_next - Q_sa)
                    self.Q[s_encoded][a_index] = new_q_value
                
                s = s_next
                s_encoded = s_next_encoded

            total_rewards.append(episode_reward)
            
            self.epsilon = max(0.01, self.epsilon * epsilon_decay) # Ensure epsilon doesn't hit zero
            
            if (episode + 1) % 100 == 0:
                print(f"     Episode {episode + 1}/{num_episodes}: Total Reward = ${episode_reward:.2f}, Epsilon = {self.epsilon:.4f}")
                
        return total_rewards
        

def fetch_prices(csv_path: str):
    prices_series = dataset_preparation.load_daily_data(csv_path)
    prices = prices_series.to_numpy()
    return prices

def evaluate_policy(agent, mdp):
    # run fully greedy policy (epsilon = 0)
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    s = mdp.initial_state()
    s_encoded = mdp.encode_state(s)

    total_reward = 0.0
    done = False

    while not done:
        action = agent.get_best_action(s_encoded)
        s_next, reward, done = mdp.step(s, action) 
        total_reward += reward
        s = s_next
        s_encoded = mdp.encode_state(s)

    agent.epsilon = old_eps
    return total_reward


def buy_and_hold(prices: np.ndarray) -> float:
    # total if we buy day 1 and hold
    return float(prices[-1] - prices[0])


if __name__ == "__main__":
    log = LOGGING or Q_LEARNING_AGENT_LOGGING
    if log: print(f"q_learning_agent.py:")

    dataset_creation.main()
    dataset_preparation.main()
    with open("data/discretized_returns.json") as f:
        disc = json.load(f)

    for csv_path, data in disc.items():
        ticker = os.path.basename(csv_path).split("_")[0]
        print(f"     ---- {ticker} ---- ")

        #load data
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        prices_full = df["Close"].to_numpy()
        cats_full = np.array(disc[csv_path]["cats"])

        # trim data, again idt this is needed 
        T = min(len(prices_full), len(cats_full))
        prices_full = prices_full[:T]
        cats_full = cats_full[:T]
        dates_full = df["Date"].to_numpy()[:T]

        train_mask = dates_full < pd.to_datetime(TRAIN_TEST_SPLIT_DATE)
        test_mask  = ~train_mask
        prices_train = prices_full[train_mask]
        cats_train   = cats_full[train_mask]
        prices_test  = prices_full[test_mask]
        cats_test    = cats_full[test_mask]
        if log: print(f"     train size: {len(prices_train)}, test size: {len(prices_test)}")

        
        mdp_train = TradingMDP(prices_train, cats_train, k=HISTORY_WINDOW_LENGTH)
        mdp_test  = TradingMDP(prices_test,  cats_test,  k=HISTORY_WINDOW_LENGTH)        
        agent = QLearningAgent(mdp_train, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

        if log: print("     ---- q-learning training ----")
        training_rewards = agent.train(num_episodes=NUM_EPISODES, epsilon_decay=EPSILON_DECAY)
        
        if log: 
            print("     ---- training complete ----")
            print(f"     states learned: {len(agent.Q)}")
            print(f"     reward in final episode: ${training_rewards[-1]:.2f}")
        

        test_reward = evaluate_policy(agent, mdp_test)
        bh_reward   = buy_and_hold(prices_test)

        print(f"     Q-learning test reward: ${test_reward:.2f}")
        print(f"     Buy & hold  test reward: ${bh_reward:.2f}")
        initial_state = mdp_test.initial_state()
        initial_s_encoded = mdp_test.encode_state(initial_state)

        if initial_s_encoded in agent.Q:
            best_action = agent.get_best_action(initial_s_encoded)
            action_name = {1: 'BUY (+1)', 0: 'HOLD (0)', -1: 'SELL (-1)'}[best_action]
            print(f"     Optimal action in first test state {initial_s_encoded}: {action_name}")
        else:
            print("     First test state was never visited during training.")