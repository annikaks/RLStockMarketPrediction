import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict
from collections import defaultdict
import random
import os
import json

from config import HISTORY_WINDOW_LENGTH, NUM_DISCRETE_CATEGORIES, ACTIONS, ALPHA, GAMMA, EPSILON, DATA_FILES, Q_LEARNING_AGENT_LOGGING, LOGGING
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
                print(f"Episode {episode + 1}/{num_episodes}: Total Reward = ${episode_reward:.2f}, Epsilon = {self.epsilon:.4f}")
                
        return total_rewards
        

def fetch_prices(csv_path: str):
    prices_series = dataset_preparation.load_daily_data(csv_path)
    prices = prices_series.to_numpy()
    return prices

if __name__ == "__main__":
    log = LOGGING or Q_LEARNING_AGENT_LOGGING
    if log: print(f"q_learning_agent.py:")

    dataset_creation.main()
    dataset_preparation.main()
    with open("data/discretized_returns.json") as f:
        disc = json.load(f)

    for csv_path, data in disc.items():
        # category / bin limits
        cats = np.array(data["cats"])
        bins = np.array(data["bins"])
        
        # raw price
        prices_series = dataset_preparation.load_daily_data(csv_path)
        prices = prices_series.to_numpy()

        # trim if needed? <- this should lwk never happen i think
        T = min(len(prices), len(cats))
        prices = prices[:T]
        cats = cats[:T]
        
        mdp = TradingMDP(prices, cats, k=HISTORY_WINDOW_LENGTH)
        agent = QLearningAgent(mdp, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

        if log: print("     ---- q-learning training ----")
        training_rewards = agent.train(num_episodes=500, epsilon_decay=0.999)
        
        if log: 
            print("     ---- training complete ----")
            print(f"     states learned: {len(agent.Q)}")
            print(f"     reward in final episode: ${training_rewards[-1]:.2f}")
        
        # --- Example Policy Check ---
        initial_state = mdp.initial_state()
        initial_s_encoded = mdp.encode_state(initial_state)

        if initial_s_encoded in agent.Q:
            best_action = agent.get_best_action(initial_s_encoded)
            action_name = {1: 'BUY (+1)', 0: 'HOLD (0)', -1: 'SELL (-1)'}[best_action]
            print(f"Optimal action learned for starting state {initial_s_encoded} is: {action_name}")
        else:
            print("Starting state was not visited enough to form a policy.")