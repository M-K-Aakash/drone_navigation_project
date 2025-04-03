import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym  # Updated gym import

# GPU Check
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
EPISODES = 1000  # Reduced from 2000 for faster training
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 128  # Increased batch size for stability
MEMORY_SIZE = 100000
TARGET_UPDATE = 5  # More frequent updates for better learning

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def train_dqn():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    q_network = DQN(state_dim, action_dim).to(DEVICE)
    target_network = DQN(state_dim, action_dim).to(DEVICE)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    memory = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON
    rewards = []

    for episode in range(EPISODES):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = torch.tensor([[random.randrange(action_dim)]], device=DEVICE)
            else:
                with torch.no_grad():
                    action = q_network(state).argmax(dim=1, keepdim=True)
            
            next_state, reward, done, _, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            memory.add(state.squeeze(0).cpu().numpy(), action.item(), reward, next_state.squeeze(0).cpu().numpy(), done)
            
            state = next_state
            total_reward += reward
            
            # Training
            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards_batch, next_states, dones = zip(*batch)
                
                states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
                actions = torch.tensor(np.array(actions).flatten(), dtype=torch.long, device=DEVICE).unsqueeze(1)
                rewards_batch = torch.tensor(np.array(rewards_batch), dtype=torch.float32, device=DEVICE)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE)
                dones = torch.tensor(np.array(dones, dtype=np.uint8), dtype=torch.float32, device=DEVICE)  # Fixed boolean conversion
                
                q_values = q_network(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = target_network(next_states).max(dim=1)[0]
                    target_q_values = rewards_batch + GAMMA * max_next_q_values * (1 - dones)
                
                loss = loss_fn(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        rewards.append(total_reward)
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Adjust Epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Print progress every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(rewards[-50:])
            print(f"Episode {episode}/{EPISODES}, Epsilon: {epsilon:.4f}, Avg Reward (last 50): {avg_reward}")

        # Early stopping if reward stabilizes
        if episode >= 100 and np.mean(rewards[-100:]) >= 195:
            print(f"Solved at Episode {episode}! Stopping early.")
            break
    
    torch.save(q_network.state_dict(), "dqn_model.pth")
    env.close()
    print("Training complete. Model saved as dqn_model.pth")

if __name__ == "__main__":
    train_dqn()