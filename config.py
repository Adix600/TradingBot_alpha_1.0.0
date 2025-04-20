API_KEY = '76jOPlKPBFIhq6EAy5gOGC6mikEEzz5q'
INITIAL_BALANCE = 10000
STATE_SIZE = 9  # Fixed to match feature vector length
ACTIONS = ['HOLD', 'BUY', 'SELL']  # Removed 'CLOSE' as unused
EPSILON = 0.1
GAMMA = 0.2
SPREAD_PCT = 0.000061  # Spread as percentage of price
MAX_PATIENCE_BONUS = 1.0

forex_symbols = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURJPY", "EURCHF", "EURGBP"
]
