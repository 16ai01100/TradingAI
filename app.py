import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import gym
from gym import spaces

# 📌 Chargement des variables d'environnement
load_dotenv()
MODEL_PATH = "models/ml_strategy_selector.pkl"
RL_MODEL_PATH = "models/rl_trading_model.zip"

# 📌 FastAPI Server
app = FastAPI()

# 📌 Simule des backtests pour entraînement
def get_backtests():
    return [
        {"strategy": "sma", "final_value": 10500, "equity_curve": np.random.randn(100).cumsum().tolist()},
        {"strategy": "rsi", "final_value": 9800, "equity_curve": np.random.randn(100).cumsum().tolist()},
        {"strategy": "ema_crossover", "final_value": 11000, "equity_curve": np.random.randn(100).cumsum().tolist()},
    ]

# 📌 Entraînement du modèle ML
def train_ml_model():
    backtests = get_backtests()
    features, labels = [], []

    for bt in backtests:
        market_features = [np.mean(bt["equity_curve"]), np.std(bt["equity_curve"]), np.min(bt["equity_curve"]), np.max(bt["equity_curve"])]
        features.append(market_features)
        labels.append(bt["strategy"])

    model = RandomForestClassifier(n_estimators=100)
    model.fit(features, labels)
    joblib.dump(model, MODEL_PATH)
    return "Modèle ML entraîné avec succès."

# 📌 Prédiction avec le modèle ML
def predict_best_strategy():
    if not os.path.exists(MODEL_PATH):
        train_ml_model()
    model = joblib.load(MODEL_PATH)
    market_features = np.random.rand(4)
    return model.predict([market_features])[0]

# 📌 Entraînement du modèle RL
class TradingEnvRL(gym.Env):
    def __init__(self, backtests):
        super().__init__()
        self.backtests = backtests
        self.current_index = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(backtests))

    def reset(self):
        self.current_index = 0
        return self._get_observation()

    def step(self, action):
        reward = self.backtests[action]["final_value"]
        self.current_index += 1
        done = self.current_index >= len(self.backtests)
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        bt = self.backtests[self.current_index]
        return np.array([np.mean(bt["equity_curve"]), np.std(bt["equity_curve"]), np.min(bt["equity_curve"]), np.max(bt["equity_curve"])])

def train_rl_model():
    backtests = get_backtests()
    env = DummyVecEnv([lambda: TradingEnvRL(backtests)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(RL_MODEL_PATH)

def predict_rl_strategy():
    if not os.path.exists(RL_MODEL_PATH):
        train_rl_model()
    model = PPO.load(RL_MODEL_PATH)
    backtests = get_backtests()
    env = DummyVecEnv([lambda: TradingEnvRL(backtests)])
    obs = env.reset()
    action, _ = model.predict(obs)
    return backtests[action[0]]["strategy"]

# 📌 APIs FastAPI
@app.get("/train_ml")
async def api_train_ml():
    return JSONResponse(content={"message": train_ml_model()})

@app.get("/choose_strategy")
async def api_choose_strategy():
    return JSONResponse(content={"best_strategy": predict_best_strategy()})

@app.get("/train_rl")
async def api_train_rl():
    train_rl_model()
    return JSONResponse(content={"message": "Modèle RL entraîné avec succès."})

@app.get("/rl_strategy")
async def api_rl_strategy():
    return JSONResponse(content={"best_strategy": predict_rl_strategy()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
