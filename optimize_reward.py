import optuna
import torch
import numpy as np
from dqn_model import DuelingNoisyDQN
from dqn_trainer import PrioritizedReplayBuffer
from historical_env import HistoricalTradingEnv
import torch.nn as nn
import torch.optim as optim
import random
import json

def calculate_custom_reward(env, action, reward, reward_gain, reward_loss, reward_hold_penalty):
    if action == 3 and env.position == 0:
        reward += reward_gain
    elif action == 4 and env.position == 0:
        reward += reward_gain
    elif env.position != 0:
        reward += reward_hold_penalty
    if reward < 0:
        reward *= reward_loss
    return reward

def objective(trial):
    env = HistoricalTradingEnv()
    input_shape = (env.window_size, 5)
    num_actions = env.action_space.n

    reward_gain = trial.suggest_float("reward_gain", 0.1, 1.0)
    reward_loss = trial.suggest_float("reward_loss", 1.0, 3.0)
    reward_hold_penalty = trial.suggest_float("reward_hold_penalty", -0.1, 0.0)

    lr = 1e-3
    gamma = 0.99
    batch_size = 64

    model = DuelingNoisyDQN(input_shape, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(10000)
    loss_fn = nn.MSELoss(reduction='none')

    total_reward = 0
    for episode in range(3):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                q_values = model(state_tensor)
            action = q_values.argmax().item() if random.random() > 0.1 else env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            reward = calculate_custom_reward(env, action, reward, reward_gain, reward_loss, reward_hold_penalty)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer.buffer) >= batch_size:
                states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float)
                dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)
                weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

                with torch.no_grad():
                    next_actions = model(next_states).argmax(1, keepdim=True)
                    next_q_values = model(next_states).gather(1, next_actions)
                    targets = rewards + gamma * next_q_values * (1 - dones)

                q_values = model(states).gather(1, actions)
                td_errors = targets - q_values
                loss = (td_errors.pow(2) * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                priorities = td_errors.abs().detach().numpy().squeeze() + 1e-6
                buffer.update_priorities(indices, priorities)

    return total_reward / 3

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial)

    with open("best_reward_config.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print("[Info] Zapisano najlepsze współczynniki do best_reward_config.json")
