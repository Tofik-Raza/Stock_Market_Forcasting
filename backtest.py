import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
DATA_PATH       = "data.npy"
ORIGINAL_PATH   = "original_df.csv"
SEQ_LEN         = 60
MODEL_LOAD_PATH = "ppo_trading.zip"

# ───────────────────────────────────────────────────────────────
# 1) Load the trained PPO model
# ───────────────────────────────────────────────────────────────
print("Loading trained PPO model...")
model = PPO.load(MODEL_LOAD_PATH)

# ───────────────────────────────────────────────────────────────
# 2) Create a fresh trading environment
# ───────────────────────────────────────────────────────────────
env = TradingEnv(
    data_path=DATA_PATH,
    original_csv=ORIGINAL_PATH,
    seq_len=SEQ_LEN,
    initial_balance=10000
)

# ───────────────────────────────────────────────────────────────
# 3) Run one episode (through entire dataset)
# ───────────────────────────────────────────────────────────────
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)

# ───────────────────────────────────────────────────────────────
# 4) Print results
# ───────────────────────────────────────────────────────────────
print(">>> BACKTEST COMPLETE <<<")
print(f"Total Profit (USD): {env.total_profit:.2f}")
print(f"Number of Trades:   {env.num_trades}")
print(f"Final Balance:      {env.balance:.2f}")
