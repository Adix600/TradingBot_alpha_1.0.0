from utils import CONFIG, log_trade, fetch_spread

sim_state = {
    "position": 0,
    "entry_price": 0.0,
    "balance": CONFIG['initial_balance'],
    "peak_balance": CONFIG['initial_balance']
}

def simulate_trade(action, price, db):
    reward = 0
    actions = ['Hold', 'Buy', 'Sell', 'Close Long', 'Close Short']
    action_label = actions[action]

    spread = fetch_spread(CONFIG['symbol'])
    risk_amount = sim_state['balance'] * CONFIG['risk_percent']

    if action == 1 and sim_state['position'] == 0:
        sim_state['position'] = 1
        sim_state['entry_price'] = price
    elif action == 2 and sim_state['position'] == 0:
        sim_state['position'] = -1
        sim_state['entry_price'] = price
    elif action == 3 and sim_state['position'] == 1:
        pnl = (price - sim_state['entry_price'] - spread)
        reward = pnl * (risk_amount / CONFIG['lot_size'])
        sim_state['balance'] += reward
        sim_state['position'] = 0
    elif action == 4 and sim_state['position'] == -1:
        pnl = (sim_state['entry_price'] - price - spread)
        reward = pnl * (risk_amount / CONFIG['lot_size'])
        sim_state['balance'] += reward
        sim_state['position'] = 0

    # aktualizacja drawdown
    sim_state['peak_balance'] = max(sim_state['peak_balance'], sim_state['balance'])
    drawdown = (sim_state['peak_balance'] - sim_state['balance']) / sim_state['peak_balance']
    if drawdown > CONFIG['max_drawdown']:
        raise RuntimeError(f"Max drawdown przekroczony: {drawdown:.2%}")

    log_trade(db, action_label, price, reward, sim_state['balance'])
    return reward