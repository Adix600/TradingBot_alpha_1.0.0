import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from config import INITIAL_BALANCE, EPSILON, GAMMA, ACTIONS, forex_symbols
from features import get_state, prepare_data, identify_strategy, get_ema200
from data_fetcher import get_data, fetch_sentiment
from database import save_trade_to_db
from utils import simulate_trade, should_close_position
from datetime import datetime, timedelta


class RLAgent(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, len(ACTIONS))
        )

    def forward(self, x):
        return self.model(x)


def train_agent(df, max_episodes=1000, timeframe='5m', reward_balance_weight=5.0):
    agent = RLAgent(input_size=len(get_state(df.iloc[0])))
    if os.path.exists("trained_agent.pth"):
        agent.load_state_dict(torch.load("trained_agent.pth"))
        print("Załadowano wytrenowany model.")
    else:
        print("Tworzenie nowego modelu.")

    opt = optim.Adam(agent.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    reward_log = []
    recent_pnls = []

    for ep in range(max_episodes):
        symbol = random.choice(forex_symbols)
        end = datetime.now() - timedelta(days=30)
        start = end - timedelta(days=365 * 20)
        base = start + timedelta(days=random.randint(0, (end - start).days - 1825))
        df = get_data(symbol, base, base + timedelta(days=365))
        sentiment = fetch_sentiment(symbol, base.strftime('%Y-%m-%d'),
                                    (base + timedelta(days=365)).strftime('%Y-%m-%d'))
        df = prepare_data(df, sentiment, symbol)
        state = get_state(df.iloc[0])
        balance, position, ptype = INITIAL_BALANCE, None, None
        steps, held, total_reward = 0, 0, 0
        actions = {a: 0 for a in ACTIONS}
        trades, wins = 0, 0
        starting_balance = balance

        for i in range(1, len(df)):
            q = agent(torch.tensor([state], dtype=torch.float32))
            eps = max(EPSILON * (0.995 ** ep), 0.01)
            a_idx = random.randint(0, 2) if random.random() < eps else q.argmax().item()
            action, price = ACTIONS[a_idx], df.iloc[i]['close']
            reward = 0

            # Logic for opening position (based on new strategy)
            if position is None and action == 'HOLD':
                steps += 1
                reward = max(-0.01 * steps, -0.1)

            elif action in ['BUY', 'SELL']:
                if position is None:
                    # EMA200 check to avoid trading in sideways markets
                    ema200 = get_ema200(symbol)
                    if (action == 'BUY' and price > ema200) or (action == 'SELL' and price < ema200):
                        position = price
                        ptype = 'long' if action == 'BUY' else 'short'
                        steps = held = 0
                        reward += min(0.05 * steps, 1.0)
                elif (
                        (ptype == 'long' and action == 'SELL') or
                        (ptype == 'short' and action == 'BUY')
                ):
                    reward, pnl = simulate_trade(position, price, ptype, held)
                    balance += pnl
                    recent_pnls.append(pnl)
                    trades += 1
                    if pnl > 0:
                        wins += 1
                    position = ptype = None
                    steps = held = 0

            elif position and should_close_position(position, price, ptype, held):
                reward, pnl = simulate_trade(position, price, ptype, held)
                balance += pnl
                trade = {
                    'time': str(df.iloc[i]['time']),
                    'action': 'adaptive_exit',
                    'entry': position,
                    'exit': price,
                    'pnl': pnl
                }
                save_trade_to_db(trade)
                recent_pnls.append(pnl)
                trades += 1
                if pnl > 0:
                    wins += 1
                position = ptype = None
                steps = held = 0

            if position:
                held += 1

            next_state = get_state(df.iloc[i])
            balance_growth = (balance - starting_balance) / starting_balance
            transaction_return = (balance - starting_balance) / (trades + 1e-5)
            target_value = (transaction_return * 0.6) + (balance_growth * reward_balance_weight * 0.4) + GAMMA * agent(
                torch.tensor([next_state], dtype=torch.float32)).max().item()
            target = torch.tensor(target_value, dtype=torch.float32)
            loss = (q[0, a_idx] - target) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()

            reward_log.append(reward)
            total_reward += reward
            actions[action] += 1
            state = next_state

        scheduler.step()  # call once per episode

        win_rate = np.mean([1 if p > 0 else 0 for p in recent_pnls]) if len(recent_pnls) >= 100 else 0
        print(
            f"\U0001F3AF Ep {ep + 1} | Trades: {trades} | Wins: {wins} | WinRate (100): {win_rate:.1%} | Balance: {balance:.2f}")
        print(f"    → Symbol: {symbol} | Okres: {base.date()} do {(base + timedelta(days=365)).date()}")
        print(f"    → Total reward: {total_reward:.2f} | Średnia nagroda na krok: {total_reward / len(df):.4f}")

        if win_rate >= 0.8:
            print("✅ Skuteczność 80% osiągnięta – koniec treningu.")
            break

    torch.save(agent.state_dict(), "trained_agent.pth")
    return agent
