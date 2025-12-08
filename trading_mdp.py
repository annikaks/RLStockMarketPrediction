from dataclasses import dataclass
from typing import Tuple
import numpy as np
from config import HISTORY_WINDOW_LENGTH, ACTIONS, INIT_CASH


@dataclass
class TradingState:
    """
    t: current time index (0 .. T-1)
    h: current position in {-1, 0, +1}  (short, flat, long) 
    """
    t: int # NOT the date, but the position in the time series (index)
    shares: int 
    cash: float

class TradingMDP:
    """
    Deterministic MDP over historical prices.

    - prices: array of floats, shape [T]
    - cats:   np.array of ints (discrete return categories), shape [T]
    - k:      window length

    State is:
      (cats[t-k+1], ..., cats[t], h_t)
    """
    def __init__(self, prices: np.ndarray, cats: np.ndarray, k: int = HISTORY_WINDOW_LENGTH, init_cash: float = INIT_CASH):
        assert len(prices) == len(cats), "      prices and cats must be same length, rerun dataset_preparation"
        self.prices = prices
        self.cats = cats
        self.T = len(prices)
        self.k = k
        self.init_cash = init_cash
        self.actions = ACTIONS

    def initial_state(self) -> TradingState:
        """
        start at t = k and flat (h = 0)
        """
        return TradingState(t=self.k, shares=0, cash=self.init_cash)

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
        if self.is_terminal(s):
            return s, 0.0, True

        t = s.t
        p_t = self.prices[t]
        p_tnext = self.prices[t + 1]

        shares = s.shares
        cash = s.cash

        # ----- apply action at price p_t -----
        if a == +1:
            if shares == 0 and cash >= p_t:
                shares = 1
                cash -= p_t
        elif a == -1:
            if shares == 1:
                shares = 0
                cash += p_t
        else:
            raise ValueError(f"Action {a} not in {{-1, +1}}")

        # ----- portfolio change from t to t+1 -----
        portfolio_t = cash + shares * p_t
        portfolio_tnext = cash + shares * p_tnext
        reward = float(portfolio_tnext - portfolio_t)

        s_next = TradingState(t=t + 1, shares=shares, cash=cash)
        done = self.is_terminal(s_next)

        # ----- shaping: penalize action opposite to price move -----
        diff = p_tnext - p_t   # actual market move
        lam = 0.05             # tune this

        if diff > 0 and a == -1:
            # market went up, agent moved "down-ish"
            reward -= lam
        elif diff < 0 and a == +1:
            # market went down, agent moved "up-ish"
            reward -= lam

        s_next = TradingState(t=t + 1, shares=shares, cash=cash)
        done = self.is_terminal(s_next)

        return s_next, reward, done

        # ==========================
        # assert a in (-1, 1, 0), "      you must -1 (sell) or 0 (hold) or +1 (buy)"

        # if self.is_terminal(s):
        #     return s, 0.0, True

        # t, h = s.t, s.h

        # p_t = self.prices[t]
        # p_tnext = self.prices[t + 1]
        # reward = h * (p_tnext - p_t)

        # h_next = int(np.clip(h + a, -1, 1))
        # s_next = TradingState(t=t + 1, h=h_next)
        # done = self.is_terminal(s_next)

        # return s_next, reward, done


    def encode_state(self, s: TradingState) -> Tuple[int, ...]:
        """
        TradingState but discrete
        Key = (cats[t-k+1], ..., cats[t], h_t)
        """
        t, shares = s.t, s.shares
        assert t >= self.k, "need k days of history for state"

        window = self.cats[t - self.k + 1 : t + 1]  # length k
        window = tuple(int(c) for c in window)

        return window + (shares,)

        # t, h = s.t, s.h
        # assert t >= self.k, "need k days of history for state"

        # window = self.cats[t - self.k + 1 : t + 1]  # k window
        # window = tuple(int(c) for c in window)

        # return window + (h,) # winodw + position 
