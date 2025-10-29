import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from Drone_env import DroneEnv
from dqn_model import DQN

# ✅ Hyperparameters
EPISODES = 200
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# ✅ Initialize environment and device
env = DroneEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# ✅ Initialize DQN
state_size = len(env.get_obs())
action_size = env.num_actions
model = DQN(state_size, action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
memory = deque(maxlen=10000)


def choose_action(state, epsilon):
    """Epsilon-greedy action selection"""
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state)
        return torch.argmax(q_values).item()


def train_step():
    """Train model from replay buffer"""
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = model(states).gather(1, actions)
    next_q = model(next_states).max(1)[0].detach()
    expected_q = rewards + (GAMMA * next_q * (1 - dones))

    loss = F.mse_loss(q_values.squeeze(), expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_model():
    """Main DQN training loop"""
    epsilon = EPSILON_START

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for t in range(200):
            action = choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            train_step()
            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Episode {ep+1}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

        if (ep + 1) % 50 == 0:
            torch.save(model.state_dict(), "drone_dqn.pth")
            print("✅ Model saved as drone_dqn.pth")

    env.close()


def test_model():
    """Test trained model"""
    model_path = "drone_dqn.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state = env.reset()
    total_reward = 0

    for t in range(200):
        action = choose_action(state, 0.0)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    print(f"✅ Test Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    # Choose whether to train or test
    mode = input("Enter 'train' to train or 'test' to evaluate model: ").strip().lower()
    if mode == "train":
        train_model()
    else:
        test_model()
