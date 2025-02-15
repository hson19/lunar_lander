import gymnasium as gym
import numpy as np
from training_udrl.training import DecisionTransformerBehavior
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
print(observation)
action_size=4
state_size = 8
model = DecisionTransformerBehavior(2,state_size,action_size,512,10,1).load("models/training_finished").to(device)
# DecisionTransformerBehavior(2,env.action_space,,512,10,1).to(device)
command = [1000,100]
for _ in range(1000):
    # this is where you would insert your policy
    # action = env.action_space.sample()
    action = np.int64(3)
    # action = np.int64(10)
    print(type(np.int64(10)))
    print(action)
    print(type(action))
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()