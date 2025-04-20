import sqlite3

def save_trade_to_db(trade):
    conn = sqlite3.connect('trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        time TEXT, action TEXT, entry REAL, exit REAL, pnl REAL)''')
    c.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?)", (
        trade['time'], trade['action'], trade['entry'], trade['exit'], trade['pnl']
    ))
    conn.commit()
    conn.close()

def log_equity_to_db(timestamp, balance, equity, position):
    conn = sqlite3.connect('trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS equity_log (
        timestamp TEXT, balance REAL, equity REAL, position TEXT)''')
    c.execute("INSERT INTO equity_log VALUES (?, ?, ?, ?)", (
        str(timestamp), balance, equity, position or 'None'
    ))
    conn.commit()
    conn.close()

def update_live_status(timestamp, price, action, position, balance, equity):
    conn = sqlite3.connect('trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS live_status (
        timestamp TEXT, price REAL, action TEXT, position TEXT, balance REAL, equity REAL)''')
    c.execute("DELETE FROM live_status")
    c.execute("INSERT INTO live_status VALUES (?, ?, ?, ?, ?, ?)", (
        str(timestamp), price, action, position or 'None', balance, equity
    ))
    conn.commit()
    conn.close()
