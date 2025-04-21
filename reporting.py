import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def export_html_report(db_path, output_file="report.html"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()

    if df.empty:
        print("[Info] Brak danych do raportu.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    total_reward = df['reward'].sum()
    avg_reward = df['reward'].mean()
    final_balance = df['balance'].iloc[-1]

    summary_html = f"""
    <h2>Podsumowanie</h2><ul>
    <li>Liczba transakcji: {df.shape[0]}</li>
    <li>Suma nagród: {total_reward:.2f}</li>
    <li>Średnia nagroda: {avg_reward:.4f}</li>
    <li>Saldo końcowe: {final_balance:.2f}</li></ul>"""

    plt.figure(figsize=(10, 4))
    df['balance'].plot(title='Saldo w czasie')
    plt.xlabel('Czas')
    plt.ylabel('Saldo')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}" />'
    buf.close()

    html = f"<html><head><title>Raport</title></head><body><h1>Raport Tradingowy</h1>{summary_html}<h2>Saldo</h2>{img_html}<h2>Log transakcji</h2>{df.to_html()}</body></html>"

    with open(output_file, "w") as f:
        f.write(html)
    print(f"[Info] Raport zapisany do {output_file}")
