import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, jsonify
from flask_cors import cross_origin
from stable_baselines3 import PPO
from trading_env import TradingEnv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
app = Flask(__name__, static_folder="static", template_folder="templates")
# CONFIGURATION
MODEL_LSTM_PATH     = "lstm.h5"
SCALER_PATH         = "scaler.pkl"
MODEL_RL_PATH       = "ppo_trading.zip"
ORIGINAL_CSV        = "original_df.csv"
DATA_NPY            = "data.npy"
SEQ_LEN             = 60
# 1) Load LSTM & scaler
lstm = tf.keras.models.load_model(MODEL_LSTM_PATH, compile=False)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
# 2) Load RL policy
policy = PPO.load(MODEL_RL_PATH)
# 3) Load original_df for latest OHLCV
original_df = pd.read_csv(ORIGINAL_CSV)
# Helper: Build state
def build_state():
    full_data = np.load(DATA_NPY)  # shape: (n_steps, 6)
    state_window = full_data[-SEQ_LEN:]
    return state_window.flatten().astype(np.float32)
# Routes
@app.route("/")
@cross_origin()
def index():
    return render_template("index.html")
@app.route("/predict")
def predict():
    state = build_state()
    action_id, _ = policy.predict(state, deterministic=True)
    action_str = {0: "hold", 1: "buy", 2: "sell"}[int(action_id)]
    return jsonify({"action": action_str, "action_id": int(action_id)})
# Run App
if __name__ == "__main__":
    app.run(debug=True)