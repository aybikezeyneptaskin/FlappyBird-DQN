import sys
sys.path.append("./PyGame-Learning-Environment")

import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done')) # transition tuple

class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([], maxlen=capacity)  
    
    #adds new transition to memory
    def push(self, *args):
        self.memory.append(transition(*args))

    # sample a batch of transitions
    def sample(self, batch_size):
        transition = random.sample(self.memory, batch_size)
        state_values = [t.state for t in transition if t is not None]
        action_values = [t.action for t in transition if t is not None]
        next_state_values = [t.next_state for t in transition if t is not None]
        reward_values = [t.reward for t in transition if t is not None]
        done_values = [t.done for t in transition if t is not None]
        #state_values = transition.state
        #action_values = transition.action

        # convert to numpy array and stack vertivallly
        state = torch.from_numpy(np.vstack(state_values)).float()
        action = torch.from_numpy(np.vstack(action_values)).long()
        next_state = torch.from_numpy(np.vstack(next_state_values)).float()
        reward = torch.from_numpy(np.vstack(reward_values)).float()
        done = torch.from_numpy(np.vstack(done_values)).float()
        return (state, action, next_state, reward, done)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_states, 128) #input, output
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, num_actions)
        
    def forward(self, x):
        #x = x.view(8, 128)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_1 = 0.5
EPS_2 = 0.1
EPS_3 = 0.01
EPS_END = 0.05
EPS_DECAY = 0.01
TAU = 0.005 #update rate
LR = 0.001 #learning rate
EPISODES = 2000

class Agent():
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.policy_net = DQN(num_states, num_actions).to(device) #instance of network
        self.target_net = DQN(num_states, num_actions).to(device) #target network is used to calculate target Q-values during training
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(10000) #insatnce of replay memory with 10000 capacity
        self.steps_done = 0

    def select_action(self, state, eps):
        #global steps_done
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        sample = random.random() #eandom value between 0 and 1 ->this value will be used to decide whether to explore or exploit
        eps_threshold = EPS_END + (eps-EPS_END) * np.exp(-1*self.steps_done*EPS_DECAY)
        
        #if random value is is greater than threshold -> exploit
        self.policy_net.eval()
        with torch.no_grad():
            actions = self.policy_net(state)
        self.policy_net.train()
        if sample > eps_threshold:
            return torch.argmax(actions).item()
        #else -> explore -> take random action
        else:
            return random.choice([0, 1])        

    def step(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done) #save transition to memory
        self.steps_done += 1
        if self.steps_done % 3 == 0: #optimize in every 3 steps
            self.optimize_model()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE) #get a abatch of samples
        state, action, next_state, reward, done = transitions
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        state_action_values = self.policy_net(state).gather(1,action) #retrieve Q values
        next_state_values = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
        expected_state_action_values = (next_state_values*GAMMA) + reward

        #compute loss
        #criterion = nn.SmoothL1Loss() 
        #loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values) # TODO: can change loss function
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

#states = p.getScreenDims() #width, height
states_dict = p.game.getGameState()
states = np.array(list(states_dict.values())) #dict to array
num_states = len(states)
actions = p.getActionSet() #[no jump, jump]
num_actions = np.array(actions).shape[0] 
print(f"actions: {actions}, states dict: {states_dict}, num actions: {num_actions}")
log_path = os.path.join('Training', 'Logs')
plt.style.use("Solarize_Light2")

def train(agent, eps, log_path=None):
    steps_done = 0 
    num_episodes = EPISODES
    episode_scores = []
    reward = 0.0

    for i in range(num_episodes):
        if p.game_over(): #check if the game is over
            p.reset_game()
        p.reset_game()
        done = False
        score = 0
        states_dict = p.game.getGameState()
        state = np.array(list(states_dict.values())) #dict to array
        while not done:
            action_index = agent.select_action(state, EPS_1)
            action = p.getActionSet()[action_index]
            reward = p.act(action)
            next_state = np.array(list(p.game.getGameState().values())) #dict to array
            agent.step(state, action_index, next_state, reward, done)
            score += reward
            state = next_state
            if(p.game_over()):
                done = True
                next_state = None
        episode_scores.append(score)
        if (i%10==0):
            print(f"Episode {i + 1}/{num_episodes}, Score: {score}")

    print(episode_scores)
    plt.plot(episode_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Episode-Score Graph for epsilon = {eps}')
    plt.savefig(os.path.join(log_path, f'episode_scores_plot_eps{eps}.png'))
    #plt.show()

    if log_path is not None:
        log_file = os.path.join(log_path, f'episode_scores_eps{eps}.log')
        with open(log_file, 'w') as f:
            for episode, score in enumerate(episode_scores):
                f.write(f"Episode {episode + 1}: {score}\n")
    
log_path_eps1 = os.path.join('Training', 'Logs', 'epsilon_0.5')
log_path_eps2 = os.path.join('Training', 'Logs', 'epsilon_0.1')
log_path_eps3 = os.path.join('Training', 'Logs', 'epsilon_0.01')

os.makedirs(log_path_eps1, exist_ok=True)
os.makedirs(log_path_eps2, exist_ok=True)
os.makedirs(log_path_eps3, exist_ok=True)

agent_eps1 = Agent(num_states=num_states, num_actions=num_actions)
train(agent_eps1, EPS_1, log_path=log_path_eps1)
torch.save(agent_eps1.policy_net.state_dict(), 'dqn_eps1.pt')
#agent_eps1.policy_net.load_state_dict(torch.load('dqn_eps1.pt'))

agent_eps2 = Agent(num_states=num_states, num_actions=num_actions)
train(agent_eps2, EPS_2, log_path=log_path_eps2)
torch.save(agent_eps2.policy_net.state_dict(), 'dqn_eps2.pt')

agent_eps3 = Agent(num_states=num_states, num_actions=num_actions)
train(agent_eps3, EPS_3, log_path=log_path_eps3)
torch.save(agent_eps3.policy_net.state_dict(), 'dqn_eps3.pt')


def plot_combined_results(log_paths, epsilons):
    all_scores = []

    for i, log_path in enumerate(log_paths):
        scores = []
        log_file = os.path.join(log_path, f'episode_scores_eps{epsilons[i]}.log')
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith("Episode"):
                    _, _, score_str = line.split()
                    scores.append(float(score_str))
        
        all_scores.append(scores)

    # Plot combined results
    for i, scores in enumerate(all_scores):
        plt.plot(scores, label=f'epsilon = {epsilons[i]}')

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Combined Episode-Score Graph')
    plt.legend()
    plt.savefig(os.path.join(log_path, f'episode_combined_scores.png'))
    plt.show()
    
log_paths = [log_path_eps1, log_path_eps2, log_path_eps3]
epsilons = [EPS_1, EPS_2, EPS_3]

plot_combined_results(log_paths, epsilons)


exit()

