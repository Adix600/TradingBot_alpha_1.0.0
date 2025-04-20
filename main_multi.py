import torch
import torch.multiprocessing as mp
import random
import numpy as np
from agent import RLAgent, train_agent
from data_fetcher import get_data, fetch_sentiment
from features import prepare_data, get_state
from config import forex_symbols
from datetime import datetime, timedelta
from utils import simulate_trade, should_close_position


def worker(rank, shared_model, lock):
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    print(f"[WORKER {rank}] uruchomiony")

    # losuj symbol i zakres danych
    symbol = random.choice(forex_symbols)
    end = datetime.now() - timedelta(days=30)
    start = end - timedelta(days=365 * 20)
    base = start + timedelta(days=random.randint(0, (end - start).days - 1825))

    # pobierz dane
    try:
        df = get_data(symbol, base, base + timedelta(days=365))
        sentiment = fetch_sentiment(symbol, base.strftime('%Y-%m-%d'), (base + timedelta(days=365)).strftime('%Y-%m-%d'))
        df = prepare_data(df, sentiment)
    except Exception as e:
        print(f"[WORKER {rank}] błąd pobierania danych: {e}")
        return

    input_size = len(get_state(df.iloc[0]))

    local_model = RLAgent(input_size)
    local_model.load_state_dict(shared_model.state_dict())

    opt = torch.optim.Adam(shared_model.parameters(), lr=0.001)

    local_model.train()
    shared_model.train()

    for ep in range(10):  # każdy worker trenuje 10 epizodów
        state = get_state(df.iloc[0])
        balance, position, ptype = 10000, None, None
        steps, held, total_reward = 0, 0, 0
        actions = {a: 0 for a in ['HOLD', 'BUY', 'SELL']}
        trades, wins = 0, 0
        starting_balance = balance

        for i in range(1, len(df)):
            q = local_model(torch.tensor([state], dtype=torch.float32))
            a_idx = q.argmax().item()
            action = ['HOLD', 'BUY', 'SELL'][a_idx]
            price = df.iloc[i]['close']
            reward = 0.0

            if position is None and action == 'HOLD':
                steps += 1
                reward = max(-0.01 * steps, -0.1)

            elif action in ['BUY', 'SELL']:
                if position is None:
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
                    trades += 1
                    if pnl > 0:
                        wins += 1
                    position = ptype = None
                    steps = held = 0

            elif position:
                atr = df.iloc[i]['atr'] if 'atr' in df.columns else None
                if should_close_position(position, price, ptype, held, atr):
                    reward, pnl = simulate_trade(position, price, ptype, held)
                    balance += pnl
                    trades += 1
                    if pnl > 0:
                        wins += 1
                    position = ptype = None
                    steps = held = 0
                else:
                    held += 1

            next_state = get_state(df.iloc[i])
            target = reward + 0.2 * local_model(torch.tensor([next_state], dtype=torch.float32)).max().item()

            loss = (q[0, a_idx] - target) ** 2

            opt.zero_grad()
            loss.backward()

            # synchroniczna aktualizacja
            with lock:
                for shared_param, local_param in zip(shared_model.parameters(), local_model.parameters()):
                    if shared_param.grad is None:
                        shared_param._grad = local_param.grad.clone()
                    else:
                        shared_param._grad += local_param.grad
                opt.step()
                local_model.load_state_dict(shared_model.state_dict())

            total_reward += reward
            actions[['HOLD', 'BUY', 'SELL'][a_idx]] += 1
            state = next_state

        win_rate = wins / trades if trades > 0 else 0
        print(f"🎯 [WORKER {rank}] Ep {ep+1} | Trades: {trades} | Wins: {wins} | WinRate: {win_rate:.1%} | Balance: {balance:.2f}")
        print(f"    → Symbol: {symbol} | Okres: {base.date()} do {(base + timedelta(days=365)).date()}")
        print(f"    → Total reward: {total_reward:.2f} | Średnia nagroda na krok: {total_reward / len(df):.4f}")


def evaluate_model(model, symbol='EURUSD'):
    from simulator import backtest
    from data_fetcher import get_data, fetch_sentiment
    from features import prepare_data
    from datetime import datetime, timedelta

    print(f"Ewaluacja modelu na symbolu {symbol}...")
    end = datetime.now()
    start = end - timedelta(days=365)
    df = get_data(symbol, start, end)
    sentiment = fetch_sentiment(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    df = prepare_data(df, sentiment)

    backtest(model, df)


def main():
    num_workers = 4
    df = get_data('EURUSD', datetime.now() - timedelta(days=365), datetime.now())
    sentiment = fetch_sentiment('EURUSD', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = prepare_data(df, sentiment)
    example_input_size = len(get_state(df.iloc[-1]))

    shared_model = RLAgent(example_input_size)
    shared_model.share_memory()

    lock = mp.Lock()
    processes = []

    for rank in range(num_workers):
        p = mp.Process(target=worker, args=(rank, shared_model, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(shared_model.state_dict(), "trained_shared_agent.pth")
    print("✅ Zapisano wspólny model po treningu wieloprocesowym.")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
