from utils import CONFIG, log_trade

sim_state = {
    "position": 0,
    "entry_price": 0.0,
    "balance": CONFIG['initial_balance']
}

def simulate_trade(action, price, db):
    reward = 0
    actions = ['Hold', 'Buy', 'Sell', 'Close Long', 'Close Short']
    action_label = actions[action]

    if action == 1 and sim_state['position'] == 0:
        sim_state['position'] = 1
        sim_state['entry_price'] = price
    elif action == 2 and sim_state['position'] == 0:
        sim_state['position'] = -1
        sim_state['entry_price'] = price
    elif action == 3 and sim_state['position'] == 1:
        reward = price - sim_state['entry_price']
        sim_state['balance'] += reward
        sim_state['position'] = 0
    elif action == 4 and sim_state['position'] == -1:
        reward = sim_state['entry_price'] - price
        sim_state['balance'] += reward
        sim_state['position'] = 0

    log_trade(db, action_label, price, reward, sim_state['balance'])
    return reward
