import subprocess
from runner import run_live_bot
from reporting import export_html_report
from utils import CONFIG, get_db_path
from transfer_tool import load_dqn_weights

def cli_menu():
    print("\n=== Forex Trading Bot CLI ===")
    print("1. Start Live Trading")
    print("2. Start Simulated Trading")
    print("3. Train DQN with Optimized Reward")
    print("4. Optimize Reward Function")
    print("5. Transfer DQN Weights to PPO")
    print("6. Generate HTML Report")
    print("7. Exit")
    return input("Select an option: ").strip()

if __name__ == "__main__":
    while True:
        try:
            choice = cli_menu()
            if choice == '1':
                CONFIG['simulate'] = False
                run_live_bot("models/ppo_model.zip")
            elif choice == '2':
                CONFIG['simulate'] = True
                run_live_bot("models/ppo_model.zip")
            elif choice == '3':
                subprocess.run(["python", "dqn_trainer.py"])
            elif choice == '4':
                subprocess.run(["python", "optimize_reward.py"])
            elif choice == '5':
                print("[Instrukcja] Użyj 'transfer_tool.py' z odpowiednią polityką PPO i DQN.")
            elif choice == '6':
                export_html_report(get_db_path())
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice.")
        except Exception as e:
            print(f"[Error] {e}")
