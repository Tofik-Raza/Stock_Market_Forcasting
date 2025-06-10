import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Trading Environment that uses:
     - data.npy → scaled features [O,H,L,C,V,Forecast] (all scaled to [0,1])
     - original_df.csv → unscaled OHLCV (for real-dollar rewards)
     
    The agent sees the last 'seq_len' rows (each of 6 features). We flatten them into a 1D vector.
    Action space: {0 = hold, 1 = buy, 2 = sell}.
    Reward is given only on a successful 'sell': profit = (sell_price - buy_price).
    The agent can hold at most 1 share at a time.
    """

    def __init__(self, data_path="data.npy", original_csv="original_df.csv", seq_len=60, initial_balance=10000):
        super(TradingEnv, self).__init__()

        # ─── Load data.npy (scaled) and original OHLCV for reward calculation ───
        self.data = np.load(data_path)  # shape: (n_steps, 6)
        self.original_df = pd.read_csv(original_csv)  # unscaled
        # Ensure 'Close' column is numeric
        self.original_df['Close'] = pd.to_numeric(self.original_df['Close'], errors='coerce')
        # Drop rows where 'Close' became NaN after coercion
        self.original_df.dropna(subset=['Close'], inplace=True)

        # Re-assert length after potentially dropping rows from original_df
        assert len(self.data) == len(self.original_df), "Mismatch between data.npy & original_df.csv lengths after cleaning."

        self.seq_len = seq_len
        self.n_steps = len(self.data)

        # ─── Action space: hold=0, buy=1, sell=2 ───
        self.action_space = spaces.Discrete(3)

        # ─── Observation space: flatten(seq_len × 6) ⇒  (seq_len * 6, ) ∈ [0,1] since every feature is scaled
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(seq_len * 6, ),
            dtype=np.float32
        )

        # ─── Trading state variables ───
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        # Reset the environment at the beginning of each episode
        self.current_step = self.seq_len  # start at index = seq_len (first possible window)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.buy_price = 0.0
        self.total_profit = 0.0
        self.num_trades = 0

        # Return the first observation
        return self._get_observation()

    def _get_observation(self):
        """Return a flattened vector of the last seq_len rows (each of 6 features)."""
        if self.current_step > self.n_steps:
             # Handle terminal state observation
             return np.zeros(self.seq_len * 6, dtype=np.float32)
        window = self.data[self.current_step - self.seq_len : self.current_step]
        return window.flatten().astype(np.float32)

    def step(self, action):
        """
        action: 0 = hold, 1 = buy, 2 = sell
        Reward only when selling (profit).
        """
        done = False
        reward = 0.0

        # Check if we are at the end of the data
        if self.current_step >= self.n_steps:
             done = True
             obs = np.zeros(self.seq_len * 6, dtype=np.float32) # Terminal observation
             # Gym environments transitioned to returning (obs, reward, terminated, truncated, info) in v0.26+
             # stable-baselines3 expects the older (obs, reward, done, info) format.
             # The shimmy wrapper should handle this, but if issues persist, you might need
             # to explicitly return terminated=done, truncated=False
             return obs, reward, done, {}


        # Get the current day’s actual (unscaled) close price
        # Ensure current_step is a valid index for original_df
        current_price = self.original_df.loc[self.current_step, "Close"]


        # ─── Execute action ───
        if action == 1:  # BUY
            # Only buy if not already holding and we have enough balance
            if self.shares_held == 0 and self.balance >= current_price:
                self.shares_held = 1
                self.buy_price = current_price
                self.balance -= current_price  # pay for one share
        elif action == 2:  # SELL
            if self.shares_held == 1:
                # Sell the one share held
                self.balance += current_price
                profit = current_price - self.buy_price
                reward = profit
                self.total_profit += profit
                self.shares_held = 0
                self.buy_price = 0.0
                self.num_trades += 1
        # else action==0: HOLD ⇒ nothing changes

        # ─── Advance to next step ───
        self.current_step += 1


        # ─── Build next observation ───
        # Check done condition again after incrementing step
        if self.current_step >= self.n_steps:
             done = True
             obs = np.zeros(self.seq_len * 6, dtype=np.float32) # Terminal observation
        else:
             obs = self._get_observation()


        return obs, reward, done, {}


    def render(self, mode="human"):
        print(
            f"Step: {self.current_step} | Balance: {self.balance:.2f} "
            f"| Shares held: {self.shares_held} | Total Profit: {self.total_profit:.2f}"
        )

    def close(self):
        pass


# import gym
# import numpy as np
# import pandas as pd
# import pickle
# from gym import spaces

# class TradingEnv(gym.Env):
#     """
#     Custom Trading Environment that uses:
#      - data.npy → scaled features [O,H,L,C,V,Forecast] (all scaled to [0,1])
#      - original_df.csv → unscaled OHLCV (for real-dollar rewards)
     
#     The agent sees the last 'seq_len' rows (each with 6 features). We flatten them into a 1D vector.
#     Action space: {0 = hold, 1 = buy, 2 = sell}.
#     Reward is given only on a successful 'sell': profit = (sell_price - buy_price).
#     The agent can hold at most 1 share at a time.
#     """

#     def __init__(self, data_path="data.npy", original_csv="original_df.csv", seq_len=60, initial_balance=10000):
#         super(TradingEnv, self).__init__()

#         # ─── Load data.npy (scaled) and original OHLCV for reward calculation ───
#         self.data = np.load(data_path)  # shape: (n_steps, 6)
#         self.original_df = pd.read_csv(original_csv)  # unscaled
#         assert len(self.data) == len(self.original_df), "Mismatch between data.npy & original_df.csv lengths."

#         self.seq_len = seq_len
#         self.n_steps = len(self.data)

#         # ─── Action space: hold=0, buy=1, sell=2 ───
#         self.action_space = spaces.Discrete(3)

#         # ─── Observation space: flatten(seq_len × 6) ⇒  (seq_len * 6, ) ∈ [0,1] since every feature is scaled
#         self.observation_space = spaces.Box(
#             low=0,
#             high=1,
#             shape=(seq_len * 6, ),
#             dtype=np.float32
#         )

#         # ─── Trading state variables ───
#         self.initial_balance = initial_balance
#         self.reset()

#     def reset(self):
#         # Reset the environment at the beginning of each episode
#         self.current_step = self.seq_len  # start at index = seq_len (first possible window)
#         self.balance = self.initial_balance
#         self.shares_held = 0
#         self.buy_price = 0.0
#         self.total_profit = 0.0
#         self.num_trades = 0

#         # Return the first observation
#         return self._get_observation()

#     def _get_observation(self):
#         """Return a flattened vector of the last seq_len rows (each of 6 features)."""
#         window = self.data[self.current_step - self.seq_len : self.current_step]
#         return window.flatten().astype(np.float32)

#     def step(self, action):
#         """
#         action: 0 = hold, 1 = buy, 2 = sell
#         Reward only when selling (profit).
#         """
#         done = False
#         reward = 0.0

#         # Get the current day’s actual (unscaled) close price
#         current_price = self.original_df.loc[self.current_step, "Close"]

#         # ─── Execute action ───
#         if action == 1:  # BUY
#             # Only buy if not already holding
#             if self.shares_held == 0:
#                 self.shares_held = 1
#                 self.buy_price = current_price
#                 self.balance -= current_price  # pay for one share
#         elif action == 2:  # SELL
#             if self.shares_held == 1:
#                 # Sell the one share held
#                 self.balance += current_price
#                 profit = current_price - self.buy_price
#                 reward = profit
#                 self.total_profit += profit
#                 self.shares_held = 0
#                 self.buy_price = 0.0
#                 self.num_trades += 1
#         # else action==0: HOLD ⇒ nothing changes

#         # ─── Advance to next step ───
#         self.current_step += 1
#         if self.current_step >= self.n_steps:
#             done = True

#         # ─── Build next observation ───
#         obs = self._get_observation() if not done else np.zeros(self.seq_len * 6, dtype=np.float32)

#         return obs, reward, done, {}

#     def render(self, mode="human"):
#         print(
#             f"Step: {self.current_step} | Balance: {self.balance:.2f} "
#             f"| Shares held: {self.shares_held} | Total Profit: {self.total_profit:.2f}"
#         )

#     def close(self):
#         pass
