import airsim
import numpy as np
import time
import gymnasium as gym


class DroneEnv:
    def __init__(self):
        # Connect to AirSim simulator
        print("Connecting to AirSim client...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("Connected!\n")

        # Take off
        self.client.takeoffAsync().join()

        # Define action space (simple movements)
        # Actions: [0: forward, 1: backward, 2: left, 3: right, 4: up, 5: down]
        self.action_space = [0, 1, 2, 3, 4, 5]
        self.num_actions = len(self.action_space)

        # Observation space: [x, y, z, vx, vy, vz]
        self.observation_space = (6,)

        # Step size for movement
        self.step_length = 2.0
        self.goal = np.array([0, 0, -10])  # Example target position

    def reset(self):
        """Reset the environment and drone position"""
        self.client.reset()
        time.sleep(1)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        state = self.get_obs()
        return state

    def get_obs(self):
        """Return current state: position + velocity"""
        kinematics = self.client.getMultirotorState().kinematics_estimated
        pos = kinematics.position
        vel = kinematics.linear_velocity
        obs = np.array([pos.x_val, pos.y_val, pos.z_val, vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        return obs

    def compute_reward(self, position):
        """Reward based on distance to goal"""
        dist = np.linalg.norm(self.goal - position)
        reward = -dist * 0.1

        # Give bonus for reaching near goal
        if dist < 2:
            reward += 100
            done = True
        else:
            done = False

        return reward, done

    def step(self, action):
        """Perform one step in the environment"""
        # Get current position
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position

        # Move drone according to chosen action
        if action == 0:   # forward
            vx, vy, vz = self.step_length, 0, 0
        elif action == 1: # backward
            vx, vy, vz = -self.step_length, 0, 0
        elif action == 2: # left
            vx, vy, vz = 0, -self.step_length, 0
        elif action == 3: # right
            vx, vy, vz = 0, self.step_length, 0
        elif action == 4: # up
            vx, vy, vz = 0, 0, -self.step_length
        elif action == 5: # down
            vx, vy, vz = 0, 0, self.step_length
        else:
            vx, vy, vz = 0, 0, 0

        # Command movement
        duration = 1.0
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()

        # Get next state
        next_state = self.get_obs()
        reward, done = self.compute_reward(next_state[:3])

        # Check for crash or out of bounds
        if abs(next_state[0]) > 100 or abs(next_state[1]) > 100 or next_state[2] > 0:
            reward -= 100
            done = True

        return next_state, reward, done, {}

    def close(self):
        """Cleanup and disarm"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("Environment closed and drone disarmed.")


# ✅ Simple test (you can run: python Drone_env.py)
if __name__ == "__main__":
    env = DroneEnv()
    obs = env.reset()
    print("Initial Observation:", obs)

    for i in range(10):
        action = np.random.choice(env.action_space)
        next_obs, reward, done, _ = env.step(action)
        print(f"Step {i+1} | Action: {action} | Reward: {reward:.2f} | Done: {done}")
        if done:
            break

    env.close()
