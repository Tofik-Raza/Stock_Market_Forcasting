import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stable_baselines3 import PPO
from trading_env import TradingEnv

app = FastAPI(title="Stock Bot Dashboard")

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
MODEL_LSTM_PATH     = "lstm.h5"
SCALER_PATH         = "scaler.pkl"
MODEL_RL_PATH       = "ppo_trading.zip"
ORIGINAL_CSV        = "original_df.csv"
DATA_NPY            = "data.npy"
SEQ_LEN             = 60

# ───────────────────────────────────────────────────────────────
# 1) Load LSTM & scaler
# ───────────────────────────────────────────────────────────────
lstm = tf.keras.models.load_model(MODEL_LSTM_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ───────────────────────────────────────────────────────────────
# 2) Load RL policy
# ───────────────────────────────────────────────────────────────
policy = PPO.load(MODEL_RL_PATH)

# ───────────────────────────────────────────────────────────────
# 3) Load original_df for latest OHLCV
# ───────────────────────────────────────────────────────────────
original_df = pd.read_csv(ORIGINAL_CSV)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def build_state():
    """
    For simplicity, we load the last SEQ_LEN rows from data.npy (which already includes the LSTM forecast).
    That array has shape (n_steps, 6), so grabbing the final SEQ_LEN rows and flattening → (SEQ_LEN*6,).
    """

    full_data = np.load(DATA_NPY)  # shape: (n_steps, 6)
    state_window = full_data[-SEQ_LEN:]
    return state_window.flatten().astype(np.float32)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict")
def predict_action():
    """
    Returns:
      {
        "action": "buy"|"hold"|"sell",
        "action_id": 0|1|2
      }
    """
    # 1) Build state from the last SEQ_LEN days
async def predict_action():
    state = build_state()
    action_id, _ = policy.predict(state, deterministic=True)
    action_str = {0: "hold", 1: "buy", 2: "sell"}[int(action_id)]
    return {"action": action_str, "action_id": int(action_id)}
