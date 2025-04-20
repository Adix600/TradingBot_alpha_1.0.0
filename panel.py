from flask import Flask, render_template, jsonify
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def get_trades():
    conn = sqlite3.connect('trades.db')
    conn.row_factory = sqlite3.Row
    trades = conn.execute("SELECT * FROM trades ORDER BY time DESC LIMIT 50").fetchall()
    conn.close()
    return trades

def get_equity_curve():
    conn = sqlite3.connect('trades.db')
    df = pd.read_sql_query("SELECT * FROM equity_log ORDER BY timestamp ASC", conn)
    conn.close()
    return df

def generate_equity_plot(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['timestamp'], df['equity'], label='Equity')
    ax.set_title("Krzywa kapitału")
    ax.set_xlabel("Czas")
    ax.set_ylabel("Kapitał")
    ax.grid(True)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def get_summary_stats():
    conn = sqlite3.connect('trades.db')
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    total = len(df)
    mean_pnl = df['pnl'].mean()
    buy = (df['action'] == 'SELL').sum()
    sl = (df['action'] == 'stop_loss').sum()
    tp = (df['action'] == 'take_profit').sum()
    return {'total': total, 'mean_pnl': mean_pnl, 'buy': buy, 'sl': sl, 'tp': tp}

def get_live_status():
    conn = sqlite3.connect('trades.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM live_status ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row

@app.route("/")
def index():
    trades = get_trades()
    equity_df = get_equity_curve()
    equity_plot = generate_equity_plot(equity_df) if not equity_df.empty else None
    stats = get_summary_stats()
    live = get_live_status()
    return render_template("index.html", trades=trades, equity_plot=equity_plot, stats=stats, live=live)

@app.route("/live")
def live():
    live = get_live_status()
    return jsonify({k: live[k] for k in live.keys()}) if live else jsonify({})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
