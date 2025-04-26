import gymnasium as gym
import logging
import numpy as np
import torch
import time
import torch.nn as nn
from torch.distributions import Categorical
from collections import namedtuple
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import wandb

from models.behavior import Behavior
from models.decision_transformer import DecisionTransformer,DecisionTransformerConfig
from models.decision_transformers_hugging import BehaviorTorchDecisionTransfomer
from models.dt import BehaviorDT
from models.utils import padding

logging.basicConfig(
    filename="logs/training_log.csv",
    level=logging.INFO,
    format="%(message)s"
)


wandb.init(project="LunarLander")
# from torchviz import make_dot
# Number of iterations in the main loop
n_main_iter = 5000

# Number of (input, target) pairs per batch used for training the behavior function
batch_size = 64

# Scaling factor for desired horizon input
horizon_scale = 0.01

# Number of episodes from the end of the replay buffer used for sampling exploratory
# commands
last_few = 75

# Learning rate for the ADAM optimizer
learning_rate = 0.03

# Number of exploratory episodes generated per step of UDRL training
n_episodes_per_iter = 100

# Number of gradient-based updates of the behavior function per step of UDRL training
n_updates_per_iter = 100

# Number of warm up episodes at the beginning of training
n_warm_up_episodes = 100

# Maximum size of the replay buffer (in episodes)
replay_size = 500

# Scaling factor for desired return input
return_scale = 20

# Evaluate the agent after `evaluate_every` iterations
evaluate_every = 2

# Target return before breaking out of the training loop
target_return = 100

# Maximun reward given by the environment
max_reward = 100000

# Maximun steps allowed
max_steps = 300

# Reward after reaching `max_steps` (punishment, hence negative reward)
max_steps_reward = -50

# Hidden units
hidden_size = 32

# backpropagate on entropy loss every `entropy_loss_every` iterations
entropy_loss_every = 10

# Times we evaluate the agent
n_evals = 1
env_name = "CartPole-v1"
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make(env_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("state size",state_size)
print("action size",action_size)
# Will stop the training when the agent gets `target_return` `n_evals` times
stop_on_solved = False
class ReplayBuffer():
    '''
    Replay buffer containing a fixed maximun number of trajectories with 
    the highest returns seen so far
    
    Params:
        size (int)
    
    Attrs:
        size (int)
        buffer (List of episodes)
    '''
    
    def __init__(self, size=0):
        self.size = size
        self.buffer = []
        
    def add(self, episode):
        '''
        Params:
            episode (namedtuple):
                (states, actions, rewards, init_command, total_return, length)
        '''
        
        self.buffer.append(episode)
    
    def get(self, num):
        '''
        Params:
            num (int):
                get the last `num` episodes from the buffer
        '''
        
        return self.buffer[-num:]
    
    def random_batch(self, batch_size):
        '''
        Params:
            batch_size (int)
        
        Returns:
            Random batch of episodes from the buffer
        '''
        
        idxs = np.random.randint(0, len(self), batch_size)
        return [self.buffer[idx] for idx in idxs]
    
    def sort(self):
        '''Keep the buffer sorted in ascending order by total return'''
        
        key_sort = lambda episode: episode.total_return
        self.buffer = sorted(self.buffer, key=key_sort)[-self.size:]
    
    def save(self, filename):
        '''Save the buffer in numpy format
        
        Param:
            filename (str)
        '''
        
        np.save(filename, self.buffer)
    
    def load(self, filename):
        '''Load a numpy format file
        
        Params:
            filename (str)
        '''
        
        raw_buffer = np.load(filename)
        self.size = len(raw_buffer)
        self.buffer = \
            [make_episode(episode[0], episode[1], episode[2], episode[3], episode[4], episode[5]) \
             for episode in raw_buffer]
    
    def __len__(self):
        '''
        Returns:
            Size of the buffer
        '''
        return len(self.buffer)


class DecisionTransformerBehavior(DecisionTransformer,Behavior):


    def init_optimizer(self,lr=0.3):
       self.optim = torch.optim.Adam(self.parameters(),lr)
       self.entropy_optim = torch.optim.Adam(self.parameters(),lr,maximize=True)
       self.scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.5, patience=10, verbose=True)
def train_behavior(behavior : Behavior, buffer : ReplayBuffer, n_updates : int, batch_size : int):
    '''Training loop
    
    Params:
        behavior (Behavior)
        buffer (ReplayBuffer)
        n_updates (int):
            how many updates we're gonna perform
        batch_size (int):
            size of the bacth we're gonna use to train on
    
    Returns:
        float -- mean loss after all the updates
    '''
    # loss = torch.nn.NLLLoss()
    loss = behavior.loss
    behavior.train()
    # torch.nn.utils.clip_grad_norm_(behavior.parameters(), max_norm=10.0)
    all_loss = []
    for update in range(n_updates):
        # params_before = {name: param.clone().detach() for name, param in behavior.named_parameters()}
        behavior.optim.zero_grad(set_to_none=False)
        # episodes = buffer.random_batch(batch_size)
        
        episodes = buffer.sort()
        episodes = buffer.get(batch_size)# get the last batch_size episodes
        batch_states,batch_actions,batch_command,batch_target = create_batches(episodes,behavior)

        batch_states = torch.stack(batch_states).to(device)
        batch_actions = torch.stack(batch_actions).to(device)
        batch_target = batch_target.squeeze().to(device,dtype=torch.int64)
        batch_command = torch.stack(batch_command).to(device)
        pred = behavior(batch_command,batch_states,batch_actions)
        
        one_hot_target = torch.nn.functional.one_hot(batch_target.to(dtype=torch.int64), num_classes=action_size).to(device)
        training_loss = loss(pred, batch_target)


        assert not torch.isnan(training_loss).any(), ('loss is a nan pred: %f and batch: %f',pred,batch_target)
        
        named_param_dict = dict(behavior.named_parameters())

        training_loss.backward()     

        # print("named_param_dict",named_param_dict['transformer.0.attention.k_net.weight'].grad)

        # print(named_param_dict['embed_timestep.weight'].grad)
        total_norm = torch.max(torch.stack([p.grad.detach().abs().max() for p in behavior.parameters()]))
        training_loss = training_loss.item()  
        behavior.optim.step()
        
        # mean_return = batch_command.mean(dim=0).mean(dim=0)
        mean_return = batch_command.mean(dim=0)
        # wandb.log({'loss':training_loss,'mean command':mean_return[0],'mean horizon':mean_return[1]})
        wandb.log({'loss':training_loss,'mean command': mean_return})

        if update % entropy_loss_every == 0 and False:
            behavior.entropy_optim.zero_grad()
            pred = behavior(batch_command,batch_states,batch_actions)
            entropy_loss = Categorical(pred).entropy().mean()
            entropy_loss.backward()
            wandb.log({'entropy_loss':entropy_loss})
            behavior.optim.step()
        all_loss.append(training_loss)
    behavior.scheduler.step(training_loss)
    print("learning rate",behavior.scheduler.get_last_lr())
    return np.mean(all_loss)

def print_episode(episodes):
    """print stats on selected episodes"""
    all_return = []
    for episode in episodes:
        all_return.append(episode.total_return)
    mean_return = np.mean(all_return)
    print(f"Total return: {mean_return}")
def create_batches(episodes,behavior):
    """
    Given a list of episodes, create three lists (batches):
        take an random number between 0 and the length of the episode - seq_len this is the start time
        take the states,actions and rewards from the start time for seq_len time steps
        add padding to the states,actions and rewards
        add the states,actions and rewards to the batch_states,batch_actions and batch_returns_to_go
    Args:
        episodes (list[Episode]): List of episode objects.
    Returns:
        tuple: (batch_states, batch_actions, batch_returns_to_go)
      """
    
    VERBOSE = 0
    batch_states = []
    batch_actions = []
    batch_returns_to_go = []
    batch_target = torch.empty( (len(episodes),1))
    
    for i,ep in enumerate(episodes):
        start_time = np.random.randint(0, max(ep.length - behavior.seq_len-1,1))
        end_time = min(ep.length-2, start_time + behavior.seq_len) # -2 because we need to have at least 1 action after the last state
        states_up_to_t = torch.tensor(ep.states[start_time:end_time]).to(device=device)

        actions_up_to_t = torch.stack([torch.nn.functional.one_hot(torch.tensor(action), num_classes=action_size).to(dtype=torch.float, device=device) for action in ep.actions[start_time:end_time]]).to(device=device)
        # command = torch.tensor(list(map(list, zip(np.flip(np.add.accumulate(np.flip(ep.rewards[start_time:]), dtype=np.float32))[:behavior.seq_len], np.arange(ep.length-start_time,0,-1) + 1))), dtype=torch.float).to(device=device)
        command = torch.tensor(
            np.flip(np.add.accumulate(np.flip(ep.rewards), dtype=np.float32))[start_time:end_time].copy(),
            dtype=torch.float
        ).to(device=device).unsqueeze(1)
       
        target = ep.actions[end_time+1]
        assert states_up_to_t.shape[0] == actions_up_to_t.shape[0] == command.shape[0], f"Shapes are not the same {states_up_to_t.shape[0]} {actions_up_to_t.shape[0]} {command.shape[0]} {behavior.seq_len}"

        if VERBOSE> 3: print("states before padding",states_up_to_t)
        states_up_to_t = padding(states_up_to_t, behavior.seq_len, behavior.state_size)
        if VERBOSE >3: print("states after padding",states_up_to_t)

        if VERBOSE>3: print("actions before padding",actions_up_to_t)
        actions_up_to_t = padding(actions_up_to_t, behavior.seq_len, behavior.action_size)
        if VERBOSE >3: print("actions after padding",actions_up_to_t)

        if VERBOSE >3: print("command before padding",command)
        command = padding(command, behavior.seq_len, behavior.r_size)
        if VERBOSE >3: print("command after padding",command)

        batch_states.append(states_up_to_t)
        batch_actions.append(actions_up_to_t)
        batch_returns_to_go.append(command)
        batch_target[i,0] = target
    
        
    assert len(batch_states) == len(batch_actions) == len(batch_returns_to_go) == len(batch_target), " Batches are not the same size {} {} {} {}".format(len(batch_states), len(batch_actions), len(batch_returns_to_go), len(batch_target))
    return batch_states, batch_actions, batch_returns_to_go, batch_target
def sample_command(buffer, last_few):
    '''Sample a exploratory command 
    
    Params:
        buffer (ReplayBuffer)
        last_few:
            how many episodes we're gonna look at to calculate 
            the desired return and horizon.
    
    Returns:
        List of float -- command
    '''
    if len(buffer) == 0: return [1, 1]
    
    # 1.
    commands = buffer.get(last_few)
    
    # 2.
    lengths = [command.length for command in commands]
    desired_horizon = round(np.mean(lengths))
    
    # 3.
    returns = [command.total_return for command in commands]
    mean_return, std_return = np.mean(returns), np.std(returns)
    desired_return = np.random.uniform(mean_return, mean_return+std_return)
    
    # return [desired_return, desired_horizon]
    return [desired_return]
def evaluate_agent(env, behavior: DecisionTransformer, command, render=False):
    '''
    Evaluate the agent performance by running an episode
    following Algorithm 2 steps
    
    Params:
        env (OpenAI Gym Environment)
        behavior (Behavior)
        command (List of float)
        render (bool) -- default False:
            will render the environment to visualize the agent performance
    '''
    behavior.eval()
    
    print('\nEvaluation.', end=' ')
        
    desired_return = command[0]
    # desired_horizon = command[1]
    init_command = torch.Tensor(command).unsqueeze(0) # T,S
    # print('Desired return: {:.2f}, Desired horizon: {:.2f}.'.format(desired_return, desired_horizon), end=' ')
    print("Desired return: {:.2f}".format(desired_return), end=' ')
    
    all_rewards = []
    test_env = gym.make(env_name, render_mode="human")
    for e in range(n_evals):
        
        done = False
        total_reward = 0
        states = torch.Tensor((test_env.reset()[0])).unsqueeze(0).unsqueeze(0)
        actions =torch.zeros([1,behavior.seq_len,behavior.action_size]) # THIS will be Filled by padding_input()
        commands = torch.Tensor(init_command).unsqueeze(0) # B,T,S
        # returns_to_go = np.array([1])
        # returns_to_go=states_to_returns_to_go(state)
        while not done:

            
            t =10  
            commands,states,actions = behavior.padding_input(commands,states,actions,device=device)
            pred_action = behavior.forward(commands.to(device),states.to(device), actions.to(device))
            pred_action = int(torch.argmax(pred_action))
            next_state, reward, done, _,_ = test_env.step(pred_action)
            time.sleep(0.1)
            one_hot = torch.nn.functional.one_hot(torch.Tensor([[pred_action]]).to(int),num_classes=action_size).to(device)
            actions=torch.cat((actions,one_hot),dim=1)
            total_reward += reward
            next_state = next_state.tolist()
            states= torch.cat((states,torch.Tensor(states)),dim=1)

            desired_return = min(desired_return - reward, max_reward)
            # desired_horizon = max(desired_horizon - 1, 1)

            # commands= torch.cat((commands,torch.Tensor([[[desired_return, desired_horizon]]]).to(device)),dim=1)
            command = torch.cat((commands,torch.Tensor([[[desired_return]]]).to(device)),dim=1)
        all_rewards.append(total_reward)
        if render: 
            test_env.close()
        
        
    
    mean_return = np.mean(all_rewards)
    print('Reward achieved: {:.2f}'.format(mean_return))
    
    behavior.train()
    
    return mean_return
make_episode = namedtuple('Episode', 
                          field_names=['states', 
                                       'actions', 
                                       'rewards', 
                                       'init_command', 
                                       'total_return', 
                                       'length', 
                                       ])    

def generate_episode(env:gym.Env, behavior:DecisionTransformerBehavior, init_command=[1, 1],random= False):
    '''
    Generate an episode using the Behaviour function.
    
    Params:
        env (OpenAI Gym Environment)
        Behaviour(func)
        init_command (List of float) -- default [1, 1]
    
    Returns:
        Namedtuple (states, actions, rewards, init_command, total_return, length)
    '''
    
    command = init_command.copy()
    # desired_return = command[0]
    # desired_horizon = command[1]
    command = [100]
    
    states = []
    states_episode = []
    actions_episode = []
    command_episode = []
    rewards = []
    
    time_steps = 0
    done = False
    total_rewards = 0
    commands = []
    state = list(env.reset()[0])
    states = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
    commands = torch.FloatTensor(command).unsqueeze(0).unsqueeze(0).to(device)
    actions = torch.zeros([1,states.shape[1],behavior.action_size]).to(device) 

    states_episode.append(state)
    command_episode.append(command)
    actions_episode.append(0)
    while not done:
        if random:
            action= np.random.randint(0,high = action_size)
            action = torch.tensor(action).to(device=device)
        else:
            commands,states,actions = behavior.padding_input(commands,states,actions,device=device)  
            with torch.no_grad():
                action=behavior(commands,states,actions)
                action= torch.argmax(action)
        returned_step = env.step(int(action.detach().to("cpu")))
        next_state, reward, done,_,_ = list(returned_step)
        actions_episode.append(int(action.detach().to("cpu")))
        states_episode.append(next_state)
        
        # Modifying a bit the reward function punishing the agent, -100, 
        # if it reaches hyperparam max_steps. The reason I'm doing this 
        # is because I noticed that the agent tents to gather points by 
        # landing the spaceshipt and getting out and back in the landing 
        # area over and over again, never switching off the engines. 
        # The longer it does that the more reward it gathers. Later on in 
        # the training it realizes that it can get more points but turning 
        # off the engines, but takes more epochs to get to that conslusion.
        if not done and time_steps > max_steps:
            done = True
            reward = max_steps_reward
        
        # Sparse rewards. Cumulative reward is delayed until the end of each episode
        #total_rewards += reward
        #reward = total_rewards if done else 0.0
        current_return = commands[0,-1,-1] 
        command = current_return - reward
        command_episode.append(command)
        rewards.append(reward)
        states_episode.append(state)

        states= torch.cat((states,torch.tensor(next_state).unsqueeze(0).unsqueeze(0).to(device)),dim=1)
        actions = torch.cat((actions,torch.nn.functional.one_hot(action,behavior.action_size).unsqueeze(0).unsqueeze(0)),dim=1)
        commands = torch.cat((commands,torch.tensor([command]).unsqueeze(0).unsqueeze(0).to(device)),dim=1)

        state = next_state.tolist()
        
        # Clipped such that it's upper-bounded by the maximum return achievable in the env
        # desired_return = min(desired_return - reward, max_reward)
        # desired_return = min(command[0] - reward, max_reward)
        
        # Make sure it's always a valid horizon
        # desired_horizon = max(desired_horizon - 1, 1)
    
        # command = [desired_return, desired_horizon]
        time_steps += 1
    return make_episode(states_episode, actions_episode, rewards, command_episode, sum(rewards), time_steps)
def UDRL(env,config, buffer=None, behavior=None, learning_history=[],render_evaluate=False):
    '''
    Upside-Down Reinforcement Learning main algrithm
    
    Params:
        env (OpenAI Gym Environment)
        buffer (ReplayBuffer):
            if not passed in, new buffer is created
        behavior (B.optimehavior):
            if not passed in, new behavior is created
        learning_history (List of dict) -- default []
    '''
    
    if behavior is None:
        behavior = initialize_behavior_function(config)
    if buffer is None:
        buffer = initialize_replay_buffer(replay_size,behavior,
                                          n_warm_up_episodes, 
                                          last_few)
    

    training_loss = []
    
    for i in range(1, n_main_iter+1):
        mean_loss = train_behavior(behavior, buffer, n_updates_per_iter, batch_size)
        training_loss.append(mean_loss)
      
        behavior.scheduler.step(mean_loss)
        
        print('Iter: {}, Loss: {}'.format(i, mean_loss))
        if (i % 100 == 0):
            behavior.save(f'weight/iter_{i}_loss_{mean_loss}')
        
        # Sample exploratory commands and generate episodes
        # generate_episodes(env, 
        #                   behavior, 
        #                   buffer, 
        #                   n_episodes_per_iter,
        #                   last_few)
        
        # if i % evaluate_every == 0:
        #     command = sample_command(buffer, last_few)
        #     mean_return = evaluate_agent(env, behavior, command,render=render_evaluate)
        #     # wandb.log({"desired return":command[0],"desired horizon":command[1],"actual return":mean_return})
        #     wandb.log({"desired return":command[0],"actual return":mean_return})
            
            # if stop_on_solved and mean_return >= target_return: 
            #     break
    
    return behavior, buffer, learning_history

def initialize_replay_buffer(replay_size,behavior, n_episodes, last_few):
    '''
    Initialize replay buffer with warm-up episodes using random actions.
    See section 2.3.1
    
    Params:
        replay_size (int)
        n_episodes (int)
        last_few (int)
    
    Returns:
        ReplayBuffer instance
        
    '''
    
    
    buffer = ReplayBuffer(replay_size)
    
    for i in range(n_episodes):
        command = sample_command(buffer, last_few)
        episode = generate_episode(env,behavior, command,random=True) # See Algorithm 2
        buffer.add(episode)
    
    buffer.sort()
    return buffer

def initialize_behavior_function(config):
    '''
    Initialize the behaviour function. See section 2.3.2
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (int) -- NOTE: not used at the moment
        learning_rate (float)
        command_scale (List of float)
    
    Returns:
        Behavior instance
    
    '''
    behavior = DecisionTransformerBehavior(config).to(device)
    
    behavior.init_optimizer(lr=torch.tensor(learning_rate))
    
    return behavior

def generate_episodes(env, behavior, buffer, n_episodes, last_few):
    '''
    1. Sample exploratory commands based on replay buffer
    2. Generate episodes using Algorithm 2 and add to replay buffer
    
    Params:
        env (OpenAI Gym Environment)
        behavior (Behavior)
        buffer (ReplayBuffer)
        n_episodes (int)
        last_few (int):
            how many episodes we use to calculate the desired return and horizon
    '''
    
    # stochastic_policy = lambda : np.random.randint(0,4)
    
    for i in range(n_episodes):
        command = sample_command(buffer, last_few)
        episode = generate_episode(env, behavior, command) # See Algorithm 2
        buffer.add(episode)
    
    # Let's keep this buffer sorted
    buffer.sort()
if __name__ == "__main__":
    DEBUG = 4
    config = DecisionTransformerConfig(state_dim=state_size,action_dim=action_size,seq_len=10,command_dim=1,n_layer=2,n_embd=32,block_size=128)
    # behavior_model = BehaviorTorchDecisionTransfomer(config)
    print(config)
    behavior_model = BehaviorDT(config)
    behavior_model.to(device)
    behavior_model.init_optimizer(lr=0.003)
    behavior_model, buffer, learning_history=UDRL(env,config,render_evaluate=True,behavior=behavior_model)
    # save the model
    behavior_model.save("training_finished")