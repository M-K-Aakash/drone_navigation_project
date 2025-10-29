import torch
import numpy as np
from Drone_env import DroneEnv
from dqn_model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# ✅ Load environment and model
env = DroneEnv()
state_size = len(env.get_obs())
action_size = env.num_actions
model = DQN(state_size, action_size).to(device)

# ✅ Load trained weights
model_path = "drone_dqn.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Run test episode
state = env.reset()
total_reward = 0

for t in range(200):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()

    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

    if done:
        break

print(f"✅ Total Test Reward: {total_reward:.2f}")
env.close()
