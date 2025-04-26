import gymnasium as gym
import numpy as np
from train import DecisionTransformerBehavior,DecisionTransformerConfig
import torch
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
print(observation)
action_size=4
state_size = 8
model = DecisionTransformerBehavior(DecisionTransformerConfig())
# DecisionTransformerBehavior(2,env.action_space,,512,10,1).to(device)
command = [1000,100] # [score,step]
command = torch.tensor(command).float().to(device)
command = command.unsqueeze(0)
model.load_state_dict(torch.load('weight/training_finished.pt'))
model.eval()
# Run the main loop
for _ in range(1000):
    # this is where you would insert your policy
    action = model.predict(observation,command)
    print(action)
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        observation = torch.tensor(observation).float().to(device)
        observation = observation.unsqueeze(0)

env.close()