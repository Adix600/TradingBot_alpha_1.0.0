# dqn_trainer.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from historical_env import HistoricalTradingEnv
from dqn_model import DuelingNoisyDQN
from utils import CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states), np.array(actions),
            np.array(rewards), np.array(next_states),
            np.array(dones), indices, weights
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

def calculate_custom_reward(env, action, reward, cfg):
    if action in (3,4) and env.position == 0:
        reward += cfg['reward_gain']
    elif env.position != 0:
        reward += cfg['reward_hold_penalty']
    if reward < 0:
        reward *= cfg['reward_loss']
    return reward

def train_dqn(
    episodes=50, batch_size=64,
    gamma=0.99, lr=1e-3, buffer_size=10000
):
    env = HistoricalTradingEnv()
    model = DuelingNoisyDQN((env.window_size,5), env.action_space.n).to(device)
    target_model = DuelingNoisyDQN((env.window_size,5), env.action_space.n).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(buffer_size)
    loss_fn = nn.MSELoss(reduction='none')

    # ładowanie optymalnych współczynników, jeśli istnieją
    cfg = {
        "reward_gain": 0.5,
        "reward_loss": 2.0,
        "reward_hold_penalty": -0.05
    }
    if os.path.exists("best_reward_config.json"):
        with open("best_reward_config.json") as f:
            cfg.update(json.load(f))
        print("[Info] Załadowano optymalne współczynniki nagrody.")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = (
                torch.tensor(state, dtype=torch.float)
                .unsqueeze(0).to(device)
            )
            with torch.no_grad():
                q_values = model(state_tensor)
            if random.random() > 0.1:
                action = q_values.argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            reward = calculate_custom_reward(env, action, reward, cfg)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer.buffer) >= batch_size:
                (
                    states, actions, rewards,
                    next_states, dones, indices, weights
                ) = buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float).to(device)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float).to(device)
                dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(device)
                weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_actions = model(next_states).argmax(1, keepdim=True)
                    next_q = target_model(next_states).gather(1, next_actions)
                    targets = rewards + gamma * next_q * (1 - dones)

                q_vals = model(states).gather(1, actions)
                td_errors = targets - q_vals
                reg = 1e-4 * sum(torch.norm(p,2) for p in model.parameters())
                loss = (td_errors.pow(2) * weights).mean() + reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                priorities = td_errors.abs().detach().cpu().numpy().squeeze() + 1e-6
                buffer.update_priorities(indices, priorities)
                model.reset_noise()
                target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode+1}/{episodes} | Total reward: {total_reward:.2f}")

    torch.save(model.state_dict(), "models/dqn_model.pth")
    print("[Done] Model zapisany jako models/dqn_model.pth")

if __name__ == '__main__':
    train_dqn()
