# RLTradeNet 🧠📈  
**A Self-Learning Forex Trading Engine Using Reinforcement Learning (LSTM + DQN)**

RLTradeNet is an open-source deep reinforcement learning (RL) framework designed to train AI agents to trade forex markets with no prior strategy or indicators. The model learns by trial and error, improving with each episode through a reward-based system. It supports multi-timeframe models and visual feedback of trades to evaluate performance.

---

## 🚀 Features

- Action-based reinforcement learning (DQN + LSTM)
- Separate models per timeframe (15M & 1H currently)
- CSV-based historical data input
- Self-learning from scratch — no initial indicators required
- Training feedback via reward graphs and trade visualizer
- Modular structure for adding new indicators or strategies dynamically
- Future-proofed for strategy stacking and DeepSeek-style reward integration

---

## 🧠 Strategy & Learning Logic

This EA uses a **Deep Q-Learning (DQN)** model enhanced with **Long Short-Term Memory (LSTM)** to learn temporal patterns in price movements. Initially, it explores randomly, assigning values (Q-values) to each action (Buy, Sell, Hold). Over time, it minimizes its regret by updating its policy through reward feedback.

We also implement **epsilon-greedy exploration**, **experience replay**, and **target networks** to ensure stable and efficient training.

---

## 📐 Mathematical Foundation

### 1. **Reinforcement Learning (DQN)**

The core learning rule is the **Bellman Equation** for Q-learning:

Q(s, a) = r + γ * max(Q(s', a'))

Where:
- `s` = current state (price, trend, volume)
- `a` = action (Buy, Sell, Hold)
- `r` = reward for taking action a in state s
- `γ` = discount factor
- `s'` = next state

### 2. **LSTM Architecture**

LSTM is used to process historical time-series data efficiently, capturing long-range dependencies. The model outputs action probabilities for each step.

### 3. **Reward System**

The reward is shaped by:
- Profit/loss from the trade
- Trade duration (penalizing long-held losing trades)
- Optional: volatility or risk-based reward scaling

Future versions will integrate a **DeepSeek-inspired reward module**, allowing multi-objective reward tracking and stacking.

---

## 🗂️ Project Structure
RLTradeNet/ ├── data/ # Historical price data (CSV format) ├── models/ # Saved models for each timeframe ├── train.py # Main training loop ├── visualize.py # Trade visualizer with matplotlib ├── environment.py # Custom ForexEnv (gym-like) ├── agent.py # DQN-LSTM agent ├── config.py # Hyperparameters └── README.md

---

## 📊 Timeframes & Setup

Currently supports:
- **15-Minute** model (`model_15m.pt`)
- **1-Hour** model (`model_1h.pt`)

Each model is trained independently using its respective dataset. Make sure you have 1-year CSV files with OHLCV format for each.

---

## 🧪 Training

```bash
# Example
python train.py --data data/EURUSD_15m.csv --tf 15m --episodes 500



