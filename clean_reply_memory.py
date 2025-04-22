import pickle
import time

def load_replay_memory(path="replay.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def save_replay_memory(memory, path="replay.pkl"):
    with open(path, "wb") as f:
        pickle.dump(memory, f)

def clean_replay_memory(memory, max_len=1000, min_reward=0.01, max_age_days=30):
    now = time.time()
    cleaned = []
    for entry in memory:
        age = (now - entry['time']) / 86400  # sekundy na dni
        if abs(entry['reward']) >= min_reward and age <= max_age_days:
            cleaned.append(entry)
    # rolling window
    cleaned = cleaned[-max_len:]
    return cleaned

if __name__ == "__main__":
    memory = load_replay_memory()
    print(f"Przed czyszczeniem: {len(memory)} wpisów")
    memory = clean_replay_memory(memory)
    save_replay_memory(memory)
    print(f"Po czyszczeniu: {len(memory)} wpisów")
