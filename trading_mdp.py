from dataclasses import dataclass
from typing import Tuple
import numpy as np
from config import WINDOW_LENGTH


@dataclass
class TradingState:
    """
    t: current time index (0 .. T-1)
    h: current position in {-1, 0, +1}  (short, flat, long) 
    """
    t: int # NOT the date, but the position in the time series (index)
    h: int # short, flat, long AT START OF DAY

class TradingMDP:
    """
    Deterministic MDP over historical prices.

    - prices: array of floats, shape [T]
    - cats:   np.array of ints (discrete return categories), shape [T]
    - k:      window length

    State is:
      (cats[t-k+1], ..., cats[t], h_t)
    """
    def __init__(self, prices: np.ndarray, cats: np.ndarray, k: int = WINDOW_LENGTH):
        assert len(prices) == len(cats), "      prices and cats must be same length, rerun dataset_preparation"
        self.prices = prices
        self.cats = cats
        self.T = len(prices)
        self.k = k

    def initial_state(self) -> TradingState:
        """
        start at t = k and flat (h = 0)
        """
        return TradingState(t=self.k, h=0)

    def is_terminal(self, s: TradingState) -> bool:
        """
        end at last day 
        """
        return s.t >= self.T - 1

    def step(self, s: TradingState, a: int) -> Tuple[TradingState, float, bool]:
        """
        Apply action {-1, +1}:

        - Reward: r_t = h_t * (p_{t+1} - p_t)
        - Position update: h_{t+1} = clip(h_t + a, -1, +1)
        - Time moves forward: t -> t+1
        """
        assert a in (-1, 1), "      you must -1 (sell) or +1 (buy)"

        if self.is_terminal(s):
            return s, 0.0, True

        t, h = s.t, s.h

        p_t = self.prices[t]
        p_tnext = self.prices[t + 1]
        reward = h * (p_tnext - p_t)

        h_next = int(np.clip(h + a, -1, 1))
        s_next = TradingState(t=t + 1, h=h_next)
        done = self.is_terminal(s_next)

        return s_next, reward, done


    def encode_state(self, s: TradingState) -> Tuple[int, ...]:
        """
        TradingState but discrete

        Key = (cats[t-k+1], ..., cats[t], h_t)
        """
        t, h = s.t, s.h
        assert t >= self.k, "need k days of history for state"

        window = self.cats[t - self.k + 1 : t + 1]  # k window
        window = tuple(int(c) for c in window)

        return window + (h,) # winodw + position 
