# Autonomous Drone Navigation using Deep Reinforcement Learning  

This project uses **Deep Reinforcement Learning (DRL)** to train a drone for **autonomous navigation and obstacle avoidance** in a simulated environment.  
This project implements Deep Reinforcement Learning for autonomous drone navigation in AirSim. The model learns obstacle avoidance and optimal pathfinding using DQN.  

 Features  
 **Deep Q-Network (DQN) / PPO-based Navigation**  
 **Obstacle Detection & Avoidance**  
 **Sensor Fusion (LiDAR, IMU, GPS)**  
 **Simulation in AirSim / ROS**  



Installation  
# Clone the Repository  
git clone https://github.com/harshraj2008/drone_navigation_project.git


cd drone_navigation_project

# Install Dependencies
Ensure you have Python 3.8+ installed, then run:
pip install -r requirements.txt

# Train the Model
Run the training script in the terminal :
python train_dqn.py
This will train the DQN agent and save the model as dqn_model.pth.

# Setup the Drone Environment
Run the script in the terminal :
python Drone_env.py
This will train the DQN agent and save the model as dqn_model.pth.

# Train the RL Model for drone environment
Run the training script in the terminal :
python drone_train.py
This will train the DQN agent and save the model as dqn_model.pth.

# After training, evaluate the model using:
python test_dqn.py







