from utils import CONFIG
from runner import run_live_bot
from reporting import export_html_report
import subprocess

def cli_menu():
    print("\n=== Forex Trading Bot CLI ===")
    print("1. Start Live Trading")
    print("2. Start Simulated Trading")
    print("3. Train LSTM Trend Model")
    print("4. Optimize PPO+LSTM with Optuna")
    print("5. Clean Replay Memory")
    print("6. Analyze Replay Memory")
    print("7. Generate HTML Report")
    print("8. Preload Replay Memory from History")
    print(f"9. Toggle Replay Memory Use [Current: {'ON' if CONFIG['use_memory'] else 'OFF'}]")
    print("10. Exit")
    return input("Select an option: ").strip()

if __name__ == "__main__":
    while True:
        try:
            choice = cli_menu()
            if choice == '1':
                CONFIG['simulate'] = False
                run_live_bot("models/ppo_lstm_model.zip")
            elif choice == '2':
                CONFIG['simulate'] = True
                run_live_bot("models/ppo_lstm_model.zip")
            elif choice == '3':
                subprocess.run(["python", "train_lstm_trend.py"])
            elif choice == '4':
                subprocess.run(["python", "optimize_ppo_lstm_advanced.py"])
            elif choice == '5':
                subprocess.run(["python", "clean_replay_memory.py"])
            elif choice == '6':
                subprocess.run(["python", "analyze_replay.py"])
            elif choice == '7':
                export_html_report("live_trades.db" if not CONFIG['simulate'] else "simulated_trades.db")
            elif choice == '8':
                subprocess.run(["python", "preload_replay_from_history.py"])
            elif choice == '9':
                CONFIG['use_memory'] = not CONFIG['use_memory']
                print(f"[✓] Replay memory set to: {'ON' if CONFIG['use_memory'] else 'OFF'}")
            elif choice == '10':
                print("Exiting.")
                break
            else:
                print("Invalid option.")
        except Exception as e:
            print(f"[Błąd] {e}")
