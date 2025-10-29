import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from Drone_env import DroneEnv  # Your custom AirSim environment
from dqn_model import DQN


EPISODES = 200
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MODEL_PATH = "drone_dqn.pth"


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


def train_model():
    print("Starting training...")
    env = DroneEnv()
    print("Connecting to AirSim client...")

    # --- Automatically determine state & action dimensions ---
    sample_state = env.reset()
    if isinstance(sample_state, (list, tuple, np.ndarray)):
        state_dim = len(sample_state)
    else:
        state_dim = 1  # fallback

    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, (list, tuple)):
        action_dim = len(env.action_space)
    else:
        raise TypeError("Unsupported action_space type in DroneEnv.")

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START

    with open("training_log.txt", "w", encoding="utf-8") as log_file:
        log_file.write("Training started...\n")

        for episode in range(EPISODES):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Epsilon-greedy action
                if random.random() < epsilon:
                    if hasattr(env.action_space, "sample"):
                        action = env.action_space.sample()
                    else:
                        action = random.randrange(action_dim)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(replay_buffer) >= BATCH_SIZE:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = (
                        states.to(device),
                        actions.to(device),
                        rewards.to(device),
                        next_states.to(device),
                        dones.to(device)
                    )

                    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = model(next_states).max(1)[0]
                    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                    loss = criterion(q_values, expected_q_values.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            log_msg = f"Episode {episode+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}\n"
            print(log_msg.strip())
            log_file.write(log_msg)
            log_file.flush()

            torch.save(model.state_dict(), MODEL_PATH)

    print("Training finished and model saved as 'drone_dqn.pth'.")


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print("Trained model found — skipping training. Use test_dqn.py for testing.")
    else:
        train_model()

