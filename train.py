import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple
import torch.nn.functional as F
from models.decision_transformer import DecisionTransformer,DecisionTransformerConfig
import logging
logging.basicConfig(
    filename="logs/training_log.csv",
    level=logging.INFO,
    format="%(message)s"
)
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
n_episodes_per_iter = 20

# Number of gradient-based updates of the behavior function per step of UDRL training
n_updates_per_iter = 100

# Number of warm up episodes at the beginning of training
n_warm_up_episodes = 10

# Maximum size of the replay buffer (in episodes)
replay_size = 500

# Scaling factor for desired return input
return_scale = 20

# Evaluate the agent after `evaluate_every` iterations
evaluate_every = 100

# Target return before breaking out of the training loop
target_return = 1

# Maximun reward given by the environment
max_reward = 1

# Maximun steps allowed
max_steps = 300

# Reward after reaching `max_steps` (punishment, hence negative reward)
max_steps_reward = -50

# Hidden units
hidden_size = 32

# Times we evaluate the agent
n_evals = 1
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
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
class Behavior(nn.Module):
    '''
    Behavour function that produces actions based on a state and command.
    NOTE: At the moment I'm fixing the amount of units and layers.
    TODO: Make hidden layers configurable.
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (int) -- NOTE: not used at the moment
        command_scale (List of float)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        
    
    def forward(self, state, command):
        '''Forward pass
        
        Params:
            state (List of float)
            command (List of float)
        
        Returns:
            FloatTensor -- action logits
        '''
        
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.command_scale)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)
    
    def action(self, state, command):
        '''
        Params:
            state (List of float)
            command (List of float)
            
        Returns:
            int -- stochastic action
        '''
        
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.sample().item()
    
    def greedy_action(self, state, command):
        '''
        Params:
            state (List of float)
            command (List of float)
            
        Returns:
            int -- greedy action
        '''
        
        logits = self.forward(state, command)
        probs = F.softmax(logits, dim=-1)
        return np.argmax(probs.detach().cpu().numpy())
    
    def init_optimizer(self, optim=Adam, lr=0.003):
        '''Initialize GD optimizer
        
        Params:
            optim (Optimizer) -- default Adam
            lr (float) -- default 0.003
        '''
        
        self.optim = optim(self.parameters(), lr=lr)
    
    def save(self, filename):
        '''Save the model's parameters
        Param:
            filename (str)
        '''
        
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        '''Load the model's parameters
        
        Params:
            filename (str)
        '''
        
        self.load_state_dict(torch.load(filename))

class DecisionTransformerBehavior(DecisionTransformer,Behavior):

    def padding_input(self,R,s,a):
        a = torch.stack([padding(elem,self.seq_len,self.action_size) for elem in a]).to(device)
        R =torch.stack([padding(elem,self.seq_len,self.r_size) for elem in R]).to(device)
        s = torch.stack([padding(elem,self.seq_len,self.state_size) for elem in s]).to(device)
        return  R,s,a 
    def init_optimizer(self,lr=0.003):
       self.optim = torch.optim.Adam(self.parameters(),lr)
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
    all_loss = []
    for update in range(n_updates):
        behavior.optim.zero_grad()
        episodes = buffer.random_batch(batch_size)
        
        batch_states = []
        batch_actions = []
        batch_target = []
        batch_return_to_go = []
        batch_command = []
        time = []
        for episode in episodes:
            T = episode.length
            t1 = np.random.randint(0, T)
            t2 = min(t1+behavior.seq_len,T-1)
            dt = t2 - t1
            dr = np.array([sum(episode.rewards[t1:t])for t in range(t2,t1,-1)])
            horizon = np.flip(np.arange(0,t2-t1)) +1.0
            command = torch.Tensor([[dr[index],horizon[index]] for index in range(horizon.shape[0])])
            
            
            st1 = torch.Tensor(episode.states[t1:t2])
            at1 = episode.actions[t1:t2+1] # take the next action with it
            at1 = torch.Tensor(at1).to(int) 
            at1 = torch.nn.functional.one_hot(at1,num_classes=action_size)
            target = at1[-1]
            at1 = at1[:-1]
            st1 = padding(st1,behavior.seq_len,behavior.state_size)
            at1 = padding(at1,behavior.seq_len,behavior.action_size)
            
            command = padding(command,behavior.seq_len,2)
            
            batch_states.append(st1)
            batch_actions.append(at1)
            batch_target.append(target)
            time.append(dt)

            batch_command.append(command)
        

        batch_states = torch.Tensor(np.array(batch_states)).to(device)
        batch_return_to_go = torch.Tensor(np.array(batch_return_to_go)).to(device)
        time = torch.Tensor(np.array(time)).to(device)
        batch_actions = torch.Tensor(np.array(batch_actions)).to(device)
        batch_target = torch.Tensor(np.array(batch_target)).to(device)
        batch_command = torch.Tensor(np.array(batch_command)).to(device)
        time = torch.Tensor(time).to(device)
        pred = behavior(batch_command,batch_states,batch_actions,time)
        
        loss = F.cross_entropy(pred, batch_target)
        assert not loss.isnan(), ('loss is a nan pred: %f and batch: %f',pred,batch_target)
        
        loss.backward()
        behavior.optim.step()
        logging.info(f'{loss.item()},{command[0]},{command[1]}')
        
        all_loss.append(loss.item())
    
    return np.mean(all_loss)
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
    
    return [desired_return, desired_horizon]
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
    desired_horizon = command[1]
    init_command = torch.Tensor(command).unsqueeze(0) # T,S
    print('Desired return: {:.2f}, Desired horizon: {:.2f}.'.format(desired_return, desired_horizon), end=' ')
    
    all_rewards = []
    
    for e in range(n_evals):
        
        done = False
        total_reward = 0
        states = torch.Tensor((env.reset()[0])).unsqueeze(0).unsqueeze(0)
        actions =torch.zeros([1,behavior.seq_len,behavior.action_size]) # THIS will be Filled by padding_input()
        commands = torch.Tensor(init_command).unsqueeze(0) # B,T,S
        # returns_to_go = np.array([1])
        # returns_to_go=states_to_returns_to_go(state)
        while not done:
            if render: env.render()
            
            t =10  
            commands,states,actions = behavior.padding_input(commands,states,actions)
            pred_action = behavior.forward(commands.to(device),states.to(device), actions.to(device),torch.Tensor(t))
            pred_action = int(torch.argmax(pred_action))
            next_state, reward, done, _,_ = env.step(pred_action)
            one_hot = torch.nn.functional.one_hot(torch.Tensor([[pred_action]]).to(int),num_classes=action_size).to(device)
            actions=torch.cat((actions,one_hot),dim=1)
            total_reward += reward
            next_state = next_state.tolist()
            states= torch.cat((states,torch.Tensor(states)),dim=1)

            desired_return = min(desired_return - reward, max_reward)
            desired_horizon = max(desired_horizon - 1, 1)

            commands= torch.cat((commands,torch.Tensor([[[desired_return, desired_horizon]]]).to(device)),dim=1)
            all_rewards.append(reward)
        if render: env.close()
        
        
    
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

def generate_episode(env, behavior:DecisionTransformerBehavior, init_command=[1, 1]):
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
    desired_return = command[0]
    desired_horizon = command[1]
    
    states = []
    actions = []
    rewards = []
    
    time_steps = 0
    done = False
    total_rewards = 0
    commands = []
    state = list(env.reset()[0])
    
    while not done:
        state_input = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        command_input = torch.FloatTensor(command).unsqueeze(0).unsqueeze(0).to(device)
        action_input = torch.zeros([1,state_input.shape[1],4]).to(device)
        command_input,state_input,action_input = behavior.padding_input(command_input,state_input,action_input)
        action=behavior(command_input,state_input,action_input,1)
        action= torch.argmax(action)
        # action = policy()
        returned_step = env.step(int(action.to('cpu')))
        next_state, reward, done,_,_ = list(returned_step)
        
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
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state.tolist()
        
        # Clipped such that it's upper-bounded by the maximum return achievable in the env
        desired_return = min(desired_return - reward, max_reward)
        
        # Make sure it's always a valid horizon
        desired_horizon = max(desired_horizon - 1, 1)
    
        command = [desired_return, desired_horizon]
        commands.append(command)
        time_steps += 1
        
    return make_episode(states, actions, rewards, commands, sum(rewards), time_steps)
def UDRL(env,config, buffer=None, behavior=None, learning_history=[],render_evaluate=False):
    '''
    Upside-Down Reinforcement Learning main algrithm
    
    Params:
        env (OpenAI Gym Environment)
        buffer (ReplayBuffer):
            if not passed in, new buffer is created
        behavior (Behavior):
            if not passed in, new behavior is created
        learning_history (List of dict) -- default []
    '''
    
    if behavior is None:
        behavior = initialize_behavior_function(config)
    if buffer is None:
        buffer = initialize_replay_buffer(replay_size,behavior,
                                          n_warm_up_episodes, 
                                          last_few)
    

    for i in range(1, n_main_iter+1):
        mean_loss = train_behavior(behavior, buffer, n_updates_per_iter, batch_size)
        
        print('Iter: {}, Loss: {}'.format(i, mean_loss), end='\r')
        if (i % 100 == 0):
            behavior.save('weigths/iter_{}_loss_{}')
        
        # Sample exploratory commands and generate episodes
        generate_episodes(env, 
                          behavior, 
                          buffer, 
                          n_episodes_per_iter,
                          last_few)
        
        if i % evaluate_every == 0:
            command = sample_command(buffer, last_few)
            mean_return = evaluate_agent(env, behavior, command,render=render_evaluate)
            
            learning_history.append({
                'training_loss': mean_loss,
                'desired_return': command[0],
                'desired_horizon': command[1],
                'actual_return': mean_return,
            })
            
            if stop_on_solved and mean_return >= target_return: 
                break
    
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
        episode = generate_episode(env,behavior, command) # See Algorithm 2
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
def padding(state, seq_len, embedding_size, device="cpu"):
    """
    Get a state and add padding if it is not the correct size.
    """
    # Handle the case where the input state is empty
    if state.nelement() == 0:  # Check for empty tensor
        return torch.zeros(seq_len, embedding_size, device=device)
    
    # Check if the state is already the correct shape
    if state.shape == (seq_len, embedding_size):
        return state.to(device)
    
    # If the sequence length is less than the target, pad at the beginning
    if state.shape[0] < seq_len:
        padding_size = seq_len - state.shape[0]
        padding_tensor = torch.zeros(padding_size, embedding_size, device=device)
        state = torch.cat((padding_tensor, state.to(device)), dim=0)
        return state
    if state.shape[0] > seq_len:
        state = state[-seq_len:]

    # Ensure the dimensions are valid
    assert state.shape[0] == seq_len, f"Expected seq_len={seq_len}, got {state.shape[0]}"
    assert state.shape[1] == embedding_size, f"Expected embedding_size={embedding_size}, got {state.shape[1]}"
    return state.to(device)
def state_to_dummy(states,n_action=4):
    """
    get a list of action returns the actions in dummy variables
    """
    new_states = []
    for state in states:
        dummy_state = np.zeros(n_action,dtype='float64')
        np.put(dummy_state,state,float(1.0),mode='raise')
        new_states.append(dummy_state)
    return new_states
if __name__ == "__main__":
    DEBUG = 4
    behavior_model, buffer, learning_history=UDRL(env,DecisionTransformerConfig(),render_evaluate=True)
    # save the model
    behavior_model.save("training_finished")