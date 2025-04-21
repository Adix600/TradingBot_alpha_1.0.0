import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def load_dqn_weights(dqn_model_path, extractor):
    dqn_weights = torch.load(dqn_model_path)
    shared_state = {
        k: v for k, v in dqn_weights.items() if k in extractor.state_dict()
    }
    extractor.load_state_dict(shared_state, strict=False)
    print("[Info] Wagi DQN załadowane do PPO feature extractor")
    return extractor

def imitation_pretrain(ppo_policy, dqn_model, env, steps=5000, batch_size=64, lr=1e-4):
    optimizer = optim.Adam(ppo_policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    all_obs = []
    all_actions = []

    obs = env.reset()
    for _ in tqdm(range(steps)):
        obs_tensor = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            dqn_output = dqn_model(obs_tensor)
            dqn_action = torch.argmax(dqn_output).item()

        all_obs.append(obs)
        all_actions.append(dqn_action)

        obs, _, done, _ = env.step(dqn_action)
        if done:
            obs = env.reset()

    print(f"[Info] Zebrano {len(all_obs)} przykładów do pretreningu")

    for epoch in range(5):
        for i in range(0, len(all_obs), batch_size):
            batch_obs = torch.tensor(all_obs[i:i+batch_size], dtype=torch.float)
            batch_actions = torch.tensor(all_actions[i:i+batch_size], dtype=torch.long)
            logits = ppo_policy.actor(batch_obs)
            loss = loss_fn(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch+1}] Imitation loss: {loss.item():.4f}")

    print("[Done] Pretrening imitacyjny PPO zakończony")
