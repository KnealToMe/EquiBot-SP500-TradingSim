import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yfinance as yf
import matplotlib.pyplot as plt


data = yf.download("^GSPC", start="2023-01-01", end="2025-01-01")
prices = data["Close"].to_numpy().reshape(-1)
dates = data.index.to_list()

# Environment
class TradingEnv(gym.Env):
    def __init__(self, prices, starting_cash=10000.0, max_position=1, fee_bps=10, gamma=1.0):
        super().__init__()
        self.prices = np.array(prices, dtype=np.float64)
        self.n = len(self.prices)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            high=np.array([np.inf, max_position, np.inf], dtype=np.float64),
            dtype=np.float64
        )
        self.starting_cash = starting_cash
        self.max_position = max_position
        self.fee_bps = fee_bps
        self.gamma = gamma
        self.reset()

    def _get_obs(self):
        price = float(self.prices[self.t])
        return np.array([price, float(self.position), float(self.cash)], dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.cash = float(self.starting_cash)
        self.equity = self.cash
        self.history = {
            "t": [],
            "price": [],
            "position": [],
            "cash": [],
            "equity": [],
            "reward": [],
            "action": []
        }
        return self._get_obs(), {}

    def step(self, action):
        price = float(self.prices[self.t])
        terminated = False
        truncated = False
        info = {}

        if action == 1 and self.position < self.max_position and self.cash >= price:
            fee = price * (self.fee_bps / 1e4)
            self.cash -= (price + fee)
            self.position += 1
        elif action == 2 and self.position > 0:
            fee = price * (self.fee_bps / 1e4)
            self.cash += (price - fee)
            self.position -= 1

        self.t += 1
        if self.t >= self.n:
            terminated = True
            self.t = self.n - 1

        new_equity = self.cash + self.position * float(self.prices[self.t])
        reward = new_equity - self.equity
        self.equity = new_equity

        
        self.history["t"].append(self.t)
        self.history["price"].append(price)
        self.history["position"].append(self.position)
        self.history["cash"].append(self.cash)
        self.history["equity"].append(self.equity)
        self.history["reward"].append(reward)
        self.history["action"].append(action)

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"t={self.t}, price={self.prices[self.t]:.2f}, pos={self.position}, cash={self.cash:.2f}, equity={self.equity:.2f}")

#Simulation
env = TradingEnv(prices)
obs, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

#boring
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(dates, prices, label="Prix", color="blue")
plt.title("Prix de clôture du S&P 500)")
plt.xlabel("Date")
plt.ylabel("Prix ($)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(env.history["t"], env.history["equity"], label="Équité", color="green")
plt.plot(env.history["t"], env.history["cash"], label="Cash", color="orange")
plt.plot(env.history["t"], env.history["position"], label="Position", color="purple")
plt.title("Évolution du portefeuille")
plt.xlabel("Étapes")
plt.ylabel("Valeur ($)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

