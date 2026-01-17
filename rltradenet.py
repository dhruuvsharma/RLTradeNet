
from dataclasses import dataclass
from collections import deque, namedtuple
import numpy as np
import pandas as pd
import random
import os
import math
import time
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    windows: List[int] = (50, 200)              
    baseline_M: int = 2000                      
    roc_L: int = 10                             
    ecd_alpha: float = 0.2                      
    lambda_div: float = 0.5                     

    initial_cash: float = 10000.0
    position_size: float = 1.0                  
    contract_multiplier: float = 1.0            
    spread_cost: float = 0.00002                
    commission: float = 0.0                     
    slippage: float = 0.0                       
    max_drawdown_limit: Optional[float] = None  

    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 200000
    min_replay_size: int = 1024
    target_update_every: int = 1000
    optimize_every: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100000
    max_grad_norm: float = 5.0
    hidden_sizes: Tuple[int,int] = (128, 128)

    num_episodes: int = 500
    max_steps_per_episode: int = 20000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeltaFeatureBuilder:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_delta(self, prices: pd.Series) -> pd.Series:
        # delta +1 / -1 / 0 per tick
        d = prices.diff().fillna(0)
        delta = pd.Series(0, index=prices.index)
        delta[d > 0] = 1
        delta[d < 0] = -1
        return delta.astype(int)

    def rolling_sum(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).sum()

    def rolling_mean_std(self, series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std().fillna(1e-8)
        return mean, std

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        assert 'price' in df.columns, "df must contain 'price' column"
        prices = df['price'].astype(float).copy()
        volumes = df['volume'] if 'volume' in df.columns else None

        delta = self.compute_delta(prices)
        out = pd.DataFrame(index=df.index)
        out['price'] = prices
        out['delta'] = delta

        for W in self.cfg.windows:
            cd = self.rolling_sum(delta, W)
            out[f'cd_{W}'] = cd

        # baseline rolling stats of the longest cd (or stack of cds)
        # We'll compute baseline stats on the longest CD series
        longest = max(self.cfg.windows)
        baseline_mean, baseline_std = self.rolling_mean_std(out[f'cd_{longest}'], self.cfg.baseline_M)

        for W in self.cfg.windows:
            out[f'z_cd_{W}'] = (out[f'cd_{W}'] - baseline_mean) / (baseline_std + 1e-9)

        # ROC
        for W in self.cfg.windows:
            out[f'roc_cd_{W}'] = out[f'cd_{W}'].diff(self.cfg.roc_L).fillna(0) / max(1, self.cfg.roc_L)

        # DV ratio if volume exists, otherwise fill 0
        if volumes is not None:
            for W in self.cfg.windows:
                vW = volumes.rolling(window=W, min_periods=1).sum()
                out[f'dv_ratio_{W}'] = out[f'cd_{W}'] / (vW + 1e-9)
        else:
            for W in self.cfg.windows:
                out[f'dv_ratio_{W}'] = 0.0

        # divergence: CD - lambda * price_change_in_ticks
        tick_size = df.get('tick_size', pd.Series(1.0, index=df.index))
        for W in self.cfg.windows:
            price_move = prices - prices.shift(W).fillna(method='bfill')
            # scale price_move by tick_size (if present)
            out[f'div_{W}'] = out[f'cd_{W}'] - self.cfg.lambda_div * (price_move / (tick_size + 1e-9))

        ecd = pd.Series(0.0, index=df.index)
        alpha = self.cfg.ecd_alpha
        last = 0.0
        for i, idx in enumerate(df.index):
            last = alpha * delta.iloc[i] + (1 - alpha) * last
            ecd.iloc[i] = last
        ecd_std = ecd.rolling(window=max(self.cfg.windows), min_periods=1).std().fillna(1e-8)
        out['ecd_norm'] = ecd / (ecd_std + 1e-9)

        # Keep price for env, and produce final "feature vector" columns
        feature_columns = [c for c in out.columns if c != 'price']
        # replace any inf/nan
        out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], 0).fillna(0.0)
        return out

class MT5Adapter:

    def __init__(self):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not available. Install it or don't use MT5Adapter.")
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    def get_ticks(self, symbol: str, from_datetime, to_datetime) -> pd.DataFrame:
        ticks = mt5.copy_ticks_range(symbol, from_datetime, to_datetime, mt5.COPY_TICKS_ALL)
        df = pd.DataFrame(ticks)

        # mt5 returns 'time' as epoch; convert
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')

        # unify price column: use 'last' if present else 'bid'/'ask' mid
        if 'last' in df.columns:
            df = df.rename(columns={'last': 'price'})
        elif 'bid' in df.columns and 'ask' in df.columns:
            df['price'] = (df['bid'] + df['ask']) / 2.0
        if 'volume' not in df.columns:
            df['volume'] = 1.0
        return df[['time', 'price', 'volume']]

class TradingEnv:
    def __init__(self, features_df: pd.DataFrame, cfg: Config,
                 start_index: Optional[int] = None, end_index: Optional[int] = None):
        self.features_df = features_df.reset_index(drop=True)
        self.cfg = cfg
        self.len = len(self.features_df)
        self.start_index = start_index if start_index is not None else 0
        self.end_index = end_index if end_index is not None else (self.len - 2)  # we read t+1
        assert self.end_index > self.start_index + 2, "Dataset too small for env"
        # derive feature columns excluding price
        self.feature_columns = [c for c in self.features_df.columns if c != 'price']
        self.n_features = len(self.feature_columns) + 1  # +1 for position

    
        # runtime state
        self._reset_state()

    def _reset_state(self):
        self.idx = self.start_index
        self.cash = self.cfg.initial_cash
        self.position = 0  # -1,0,1
        self.entry_price = 0.0
        self.prev_equity = self.cash
        self.done = False
        self.max_equity = self.cash
        self.episode_steps = 0

    def reset(self, random_start: bool = True):
        if random_start:
            self.idx = random.randint(self.start_index, max(self.start_index, self.end_index - 1000))
        else:
            self.idx = self.start_index
        self.cash = self.cfg.initial_cash
        self.position = 0
        self.entry_price = 0.0
        self.prev_equity = self.cash
        self.max_equity = self.cash
        self.done = False
        self.episode_steps = 0
        return self._get_obs()

    def _get_obs(self):
        row = self.features_df.iloc[self.idx]
        feats = row[self.feature_columns].values.astype(np.float32)
        pos_val = np.float32(self.position)
        obs = np.concatenate([feats, [pos_val]])
        return obs

    def _current_price(self):
        return float(self.features_df.iloc[self.idx]['price'])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on terminated env. Call reset().")

        desired_pos = -1 if action == 0 else (0 if action == 1 else 1)
        price_t = float(self.features_df.iloc[self.idx]['price'])

        # trading costs modeled as a fixed spread + commission + slippage
        trade_cost = 0.0
        position_changed = (desired_pos != self.position)
        if position_changed:
            # close existing position (if any) at price_t
            if self.position != 0:
                realized = (price_t - self.entry_price) * self.position * self.cfg.position_size * self.cfg.contract_multiplier
                self.cash += realized
            # apply transaction cost for closing & opening
            trade_cost = (abs(desired_pos - self.position)) * (self.cfg.spread_cost + self.cfg.commission + abs(self.cfg.slippage * price_t))
            # open new position if desired_pos != 0
            if desired_pos != 0:
                # set entry price at price_t plus half-spread/slippage if you want - simple model: add slippage
                self.entry_price = price_t * (1.0 + self.cfg.slippage if desired_pos == 1 else 1.0 - self.cfg.slippage)
            else:
                self.entry_price = 0.0
            self.position = desired_pos
            # subtract trade_cost from cash (assume immediate cost)
            self.cash -= trade_cost

        # advance time one tick
        # We compute mark-to-market reward based on move to next tick
        next_idx = min(self.idx + 1, self.end_index)
        price_next = float(self.features_df.iloc[next_idx]['price'])
        # mark-to-market PnL
        unrealized = 0.0
        if self.position != 0:
            unrealized = (price_next - self.entry_price) * self.position * self.cfg.position_size * self.cfg.contract_multiplier
        equity = self.cash + unrealized
        reward = equity - self.prev_equity
       
       
        self.prev_equity = equity
        self.max_equity = max(self.max_equity, equity)
        # check drawdown kill
        if (self.cfg.max_drawdown_limit is not None) and ((self.max_equity - equity) / (self.max_equity + 1e-9) > self.cfg.max_drawdown_limit):
            self.done = True

            
        # move index
        self.idx = next_idx
        self.episode_steps += 1
        if self.idx >= self.end_index or self.episode_steps >= self.cfg.max_steps_per_episode:
            self.done = True

        return self._get_obs(), float(reward), bool(self.done), {
            'equity': float(equity),
            'cash': float(self.cash),
            'position': int(self.position),
            'price_next': price_next,
            'trade_cost': float(trade_cost)
        }

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int,int]=(128,128)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_sizes=cfg.hidden_sizes).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_sizes=cfg.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.action_dim = action_dim
        self.steps_done = 0
        self.eps_start = cfg.eps_start
        self.eps_end = cfg.eps_end
        self.eps_decay_steps = cfg.eps_decay_steps
        # copy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * max(0.0, (1.0 - self.steps_done / float(self.eps_decay_steps)))
        self.steps_done += 1
        if eval_mode or random.random() > eps_threshold:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                qvals = self.policy_net(s)
                action = int(qvals.argmax(dim=1).item())
                return action
        else:
            return random.randrange(self.action_dim)

    def push_transition(self, *args):
        self.replay.push(*args)

    def optimize(self):
        if len(self.replay) < self.cfg.min_replay_size:
            return None

        if len(self.replay) < self.cfg.batch_size:
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        batch = Transition(*zip(*batch))
        # convert to tensors
        state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # current Q-values
        q_values = self.policy_net(state_batch).gather(1, action_batch)  # (B,1)

        # Double DQN: actions for next_state from policy_net, values from target_net
        next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)  # (B,1)
        next_q_values = self.target_net(next_state_batch).gather(1, next_actions).detach()  # (B,1)

        expected_q = reward_batch + (1.0 - done_batch) * (self.cfg.gamma * next_q_values)

        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg': self.cfg
        }, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(d['policy_state'])
        self.target_net.load_state_dict(d['target_state'])
        self.optimizer.load_state_dict(d['optimizer_state'])

def train_agent(env: TradingEnv, agent: DQNAgent, cfg: Config, verbose: bool = True):
    rewards_history = []
    best_eval = -np.inf
    total_steps = 0
    for ep in range(cfg.num_episodes):
        state = env.reset(random_start=True)
        ep_reward = 0.0
        losses = []
        done = False
        steps = 0
        while not done:
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, done, info = env.step(action)
            agent.push_transition(state, action, reward, next_state, done)
            if total_steps % cfg.optimize_every == 0:
                loss = agent.optimize()
                if loss is not None:
                    losses.append(loss)
            if total_steps % cfg.target_update_every == 0 and total_steps > 0:
                agent.update_target()
            state = next_state
            ep_reward += reward
            total_steps += 1
            steps += 1
        rewards_history.append(ep_reward)
        if verbose and (ep % max(1, cfg.num_episodes // 20) == 0):
            mean_r = np.mean(rewards_history[-20:]) if len(rewards_history) >= 1 else ep_reward
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            print(f"[Train] Ep {ep:4d}/{cfg.num_episodes}  steps:{steps:5d} ep_reward:{ep_reward:8.2f} recent_mean:{mean_r:8.2f} loss:{avg_loss:.6f}")
    return rewards_history

def evaluate_agent(env: TradingEnv, agent: DQNAgent, episodes: int = 5):
    rewards = []
    for ep in range(episodes):
        state = env.reset(random_start=False)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
    return rewards

def generate_synthetic_ticks(length: int = 50000, start_price: float = 1.1000, vol: float = 1e-4):

    rng = np.random.default_rng(12345)
    prices = np.empty(length)
    prices[0] = start_price
    volumes = np.abs(rng.normal(loc=1.0, scale=0.5, size=length)) + 0.1
    trend = 0.0
    for i in range(1, length):
        if rng.random() < 0.001:
            trend = rng.normal(loc=0.0, scale=2e-4)
        shock = rng.normal(loc=trend, scale=vol)
        prices[i] = max(1e-6, prices[i-1] + shock)
    df = pd.DataFrame({'price': prices, 'volume': volumes})
    return df

if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    print("Device:", cfg.device)

    df_ticks = generate_synthetic_ticks(length=20000)

    fb = DeltaFeatureBuilder(cfg)
    feat_df = fb.compute_features(df_ticks)

    env = TradingEnv(feat_df, cfg)


    state_dim = env.n_features
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim, cfg)
    print("Warming up replay buffer with random actions...")
    for _ in range(cfg.min_replay_size):
        s = env.reset(random_start=True)
        a = random.randrange(action_dim)
        ns, r, d, _ = env.step(a)
        agent.push_transition(s, a, r, ns, d)
        if d:
            env.reset()

    print("Starting training...")
    train_agent(env, agent, cfg, verbose=True)
    print("Evaluating agent...")
    eval_rewards = evaluate_agent(env, agent, episodes=5)
    print("Eval rewards:", eval_rewards)
    model_path = "models/dqn_delta_trader.pth"
    print("Saving model to", model_path)
    agent.save(model_path)
