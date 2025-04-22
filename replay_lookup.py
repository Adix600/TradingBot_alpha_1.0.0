import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_replay_memory(path="replay.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def save_replay_memory(memory, path="replay.pkl"):
    with open(path, "wb") as f:
        pickle.dump(memory, f)

def find_similar_case(current_obs, memory, threshold=0.95):
    if not memory:
        return False, None

    current_obs_flat = current_obs.flatten().reshape(1, -1)
    candidates = [case['state'].flatten() for case in memory]
    similarities = cosine_similarity(current_obs_flat, candidates)[0]
    max_idx = int(np.argmax(similarities))
    if similarities[max_idx] >= threshold:
        return True, memory[max_idx]
    return False, None
