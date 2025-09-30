import numpy as np
import gym
from gym import spaces

class MarketEnv(gym.Env):
    """
    Custom Market Environment for RL trading.
    Discrete action space: 0=Hold, 1=Buy, 2=Sell, 3=Close
    Observations: price window (OHLCV)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        window_size=50,
        initial_balance=10000,
        spread=0.0002,
        slippage=0.0001,
        max_drawdown=0.2,
        daily_loss_limit=0.05,
    ):
        super(MarketEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.spread = spread
        self.slippage = slippage
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit

        # RL action/observation spaces
        self.action_space = spaces.Discrete(4)  # Hold, Buy, Sell, Close
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32
        )

        # Episode variables
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.max_equity = self.initial_balance
        self.position = None  # "long" / "short" / None
        self.entry_price = 0.0
        self.current_step = self.window_size
        self.done = False
        self.episode_return = 0.0
        return self._get_observation()

    def _get_observation(self):
        return self.df.iloc[self.current_step - self.window_size : self.current_step].values

    def _execute_trade(self, action):
        price = self.df.loc[self.current_step, "Close"]
        next_open = self.df.loc[self.current_step, "Open"]

        # Simulated slippage & spread
        trade_price = next_open * (1 + np.random.uniform(-self.slippage, self.slippage))
        if action == 1:  # Buy
            trade_price += self.spread
        elif action == 2:  # Sell
            trade_price -= self.spread

        reward = 0.0

        if action == 1:  # Buy
            if self.position is None:
                self.position = "long"
                self.entry_price = trade_price
            elif self.position == "short":
                # Close short → PnL
                reward = (self.entry_price - trade_price) * 100
                self.balance += reward
                self.position, self.entry_price = None, 0.0

        elif action == 2:  # Sell
            if self.position is None:
                self.position = "short"
                self.entry_price = trade_price
            elif self.position == "long":
                # Close long → PnL
                reward = (trade_price - self.entry_price) * 100
                self.balance += reward
                self.position, self.entry_price = None, 0.0

        elif action == 3:  # Close
            if self.position == "long":
                reward = (trade_price - self.entry_price) * 100
                self.balance += reward
            elif self.position == "short":
                reward = (self.entry_price - trade_price) * 100
                self.balance += reward
            self.position, self.entry_price = None, 0.0

        # Update equity
        self.equity = self.balance
        if self.position == "long":
            self.equity += (price - self.entry_price) * 100
        elif self.position == "short":
            self.equity += (self.entry_price - price) * 100

        self.max_equity = max(self.max_equity, self.equity)
        self.episode_return += reward
        return reward

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, {}

        reward = self._execute_trade(action)
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.done = True

        # Risk management termination
        drawdown = 1 - self.equity / self.max_equity
        if drawdown >= self.max_drawdown:
            self.done = True

        # Daily loss cap
        if (self.equity - self.initial_balance) / self.initial_balance <= -self.daily_loss_limit:
            self.done = True

        # Penalties
        reward -= 0.001 * abs(self.equity - self.balance)  # position penalty
        reward -= 0.0005 * drawdown  # drawdown penalty

        obs = self._get_observation()
        return obs, reward, self.done, {}

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"Equity: {self.equity:.2f}, Position: {self.position}"
        )


if __name__ == "__main__":
    # quick test with synthetic OHLCV
    n = 500
    dates = np.arange(n)
    prices = np.cumsum(np.random.randn(n)) + 100
    df = {
        "Open": prices + np.random.randn(n) * 0.1,
        "High": prices + np.random.rand(n) * 0.2,
        "Low": prices - np.random.rand(n) * 0.2,
        "Close": prices,
        "Volume": np.random.randint(100, 1000, size=n),
    }
    import pandas as pd
    df = pd.DataFrame(df)

    env = MarketEnv(df)
    obs = env.reset()
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, d, _ = env.step(a)
        env.render()
        if d:
            break