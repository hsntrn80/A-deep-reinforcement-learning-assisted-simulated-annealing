import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T    

# save experiences in replay memory
# Example : 
#  e = Experience(2,3,1,4) 
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


# Policy Network to be trained

class DQN(nn.Module):
    def __init__(self, n_skus, n_actinos):
        super().__init__()
        nh1 = round(math.sqrt((n_actinos+2)*n_skus)+2*math.sqrt(n_skus/n_actinos))
        nh2 = round(n_actinos*math.sqrt(n_skus/(n_actinos+2))) 
        self.fc1 = nn.Linear(in_features=n_skus, out_features=nh1)   
        self.fc2 = nn.Linear(in_features=nh1, out_features=nh2)
        self.out = nn.Linear(in_features=nh2, out_features=n_actinos)
        # self.sm = nn.Softmax(dim=1)

    # gets SKU as input and returns n_actions as output
    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

# RL strategy
class EpsilonGreedyStrategy():
    # exploration rate epsilon will exponentially change over time
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.actions = torch.tensor(range(num_actions))

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore      
        else: 
            with torch.no_grad():                
                # preds = policy_net(torch.FloatTensor(state))[0]
                # idx = preds.multinomial(num_samples=1, replacement=True)
                # return self.actions[idx].to(self.device) # exploit   
                return policy_net(torch.FloatTensor(state)).argmax(dim=1).to(self.device) # exploit  


class EnvironmentManager():
    def __init__(self, device, S, min_cluster, max_cluster, tc_func):
        self.device = device
        self.n_actions_available = 3
        self.min_cluster = min_cluster
        self.max_cluster = max_cluster
        self.tc_func = tc_func
        self.setState(S)

    def setState(self, S):
        self.S = S
        self.S_initial = S
        self.S_prev = S # Previous solution
        self.prev_cost = self.tc_func(S)[0]  # reward is the difference between new solutions cost minus previous solutions cost                

    def reset(self):
        self.S = self.S_initial
        self.S_prev = self.S
        self.prev_cost = self.tc_func(self.S)[0]

    def ns_mutate_random(self):
        """ Changes value of a random element of x. 
            The new values are between min_cluster and max_cluster inclusive. """
        s_new = self.S[:]
        n_skus = len(s_new)
        idx = random.randint(0, n_skus-1)
        ex_cluster_number = self.S[idx]
        numbers = list(range(self.min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, self.max_cluster+1))
        s_new[idx] = random.choice(numbers)
        return s_new

    def ns_mutate_random2(self):
        """ Changes values of two random elements of x.
            The new values are between min_cluster and max_cluster inclusive. """
        s_new = self.S[:]
        n_skus = len(s_new)
        idx1, idx2 = random.sample(range(n_skus), 2)
        ex_cluster_number = self.S[idx1]
        numbers = list(range(self.min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, self.max_cluster+1))
        s_new[idx1] = random.choice(numbers)
        ex_cluster_number = self.S[idx2]
        numbers = list(range(self.min_cluster, ex_cluster_number)) + list(range(ex_cluster_number + 1, self.max_cluster+1))
        s_new[idx2] = random.choice(numbers)        
        return s_new     

    def ns_two_way_swap(self):
        """ Randomly swaps two elements of x. """
        s_new = self.S[:]
        n_skus = len(s_new)
        idx1, idx2 = random.sample(range(n_skus), 2)
        s_new[idx1] = self.S[idx2]
        s_new[idx2] = self.S[idx1]
        return s_new 

    def num_actions_available(self):
        return self.n_actions_available

    def take_action(self, action):        
        if action == 0:
            S_new = self.ns_mutate_random()
        elif action == 1:
            S_new = self.ns_mutate_random2()
        else:
            S_new =self.ns_two_way_swap()
        self.S_prev = self.S
        self.S = S_new
        new_cost = self.tc_func(S_new)[0]
        reward = self.prev_cost-new_cost  # new cost smaller? higher reward
        self.prev_cost = new_cost
        return new_cost, torch.FloatTensor([reward], device=self.device)

    def get_state(self):
        return  [self.S]  # add the batch dimension


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    print("Episode", len(values), "\n", \
    moving_avg_period, "episode moving avg:", moving_avg[-1])
    plt.show()


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

# plot(np.random.rand(300), 100)


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)



class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod        
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'