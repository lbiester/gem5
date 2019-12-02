import random
import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import bernoulli
from collections import namedtuple
from pb_DQN import PrefetchBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class FNN(nn.Module):
    def __int__(self, input_size, hidden_sizes, output_size):
        super(FNN, self).__init__()
        # for now, assuming 2 hidden layers
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                nn.ReLU(),
                                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                nn.ReLU(),
                                nn.Linear(hidden_sizes[1], output_size))

    def forward(self, x):
        x = self.fc(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden


class DQNPrefetcher:
    def __init__(self, pcs, address_diffs, use_window=128, batch_size=128, gamma=0.999, eps_start=0.9, 
                 eps_end=0.05, eps_decay=200, target_update=1000, net_type='FNN', hidden_size=512):
        if sys.version_info[0] != 3:
            raise Exception("RL Prefetcher must be used with python 3!")

        self.pcs = pcs
        self.address_diffs = address_diffs
        self.pc2int = {pc: i for i, pc in enumerate(self.pcs)}
        # TODO: could maybe represnt diffs in way that accounts for their value
        self.diff2int = {ad: i for i, ad in enumerate(self.address_diffs)}
        self.epsilon = epsilon
        # store history of actions/addresses chosen and the state and timestep in which they were chosen
        self.choice_history_buffer = PrefetchBuffer(use_window)
        self.use_window = use_window
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.net_type = net_type
        self.state_dim = len(pcs) + len(address_diffs)
        self.action_dim = len(address_diffs)
        if self.net_type == 'FNN':
            self.policy_net = FNN(self.state_dim, [hidden_size, hidden_size], self.action_dim).to(device)
            self.target_net = FNN(self.state_dim, [hidden_size, hidden_size], self.action_dim).to(device)
        else:
            self.policy_net = RNN(self.state_dim, hidden_size, self.action_dim)
            self.target_net = RNN(self.state_dim, hidden_size, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def train_step(self, curr_address):
        # assign rewards to "stale" items that haven't been given rewards yet and add to replay memory
        stale_item = self.choice_history_buffer.get_stale_item()
        if stale_item is not None:
            if stale_item.reward is None:
                stale_item.reward = -1
                next_item = self.choice_history_buffer.get_next_pbi(stale_item)
                self.memory.push(stale_item.state, stale_item.action, next_item.state, stale_item.reward)
        
        # assign rewards to correct preftech choice and add to replay memory
        causal_prefetch_item = self.choice_history_buffer.get_causal_prefetch_item(curr_address)
        if causal_prefetch_item is not None:
            delay = self.choice_history_buffer.step - causal_prefetch_item.step
            reward = (self.use_window - delay) / self.use_window
            next_item = self.choice_history_buffer.get_next_pbi(causal_preftech_item)
            casual_prefetch_item.reward = reward
            self.memory.push(causal_prefetch_item.state, causal_prefetch_item.action, next_item.state, causal_prefetch_item.reward)

        self.choice_history_buffer.remove_stale_item()

        self.optimize_model()
        
        if self.steps_done % self.target_update == 0:
            target_net.load_state_dict(policy_next.state_dict())

    
    def select_action(self, curr_state):
        self.train_step(curr_state[0])
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        explore = bernoulli.rvs(eps)
        valid_action_ids = self._get_valid_action_ids(self.address_diffs, curr_state)
        if explore:
            action_id = random.choice(valid_action_ids)
        else:
            # transform curr_state into tensor
            ad_idx = self.diff2int[curr_state[0]]
            pc_idx = self.pc2int[curr_state[1]]
            state_tensor = torch.cat((F.one_hot(ad_idx, len(self.address_diffs)), F.one_hot(pc_idx, self.pcs)), 1)
            # restrict to just valid actions
            # get max action out of all valid actions
            with torch.no_grad():
                action_id = self.policy_net(state_tensor)[valid_action_ids].max(1)[1].view(1,1)
        
        address_diff = self.address_diffs[action_id]
        address = curr_state[0] + address_diff
        
        self.choice_history_buffer.add_item(address_diff, address, curr_state)

        return address

    def _get_valid_action_ids(self, address_diffs, curr_state):
        return [i for i, diff in enumerate(address_diffs) if self._is_valid_action(diff, curr_state)]

    def _is_valid_action(self, address_diff, curr_state):
        address = curr_state[0] + address_diff
        # want to make sure not to select the same address for pre-fetching twice before it is used
        # also want to make sure we are not pre-fetching the current address (it will be cached anyway)
        # also that address is within address space
        return address not in self.choice_history_buffer and address_diff != 0 and address > 0 and address < 2 ** 64

    def optimize_model():
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # convert batch array of tranistions into transition of batch-arrays
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # compute Q(s_t, a) --> model computes Q(s_t) so then compute columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # compute V(s_t+1) for all next states
        # expected values of actions are computed based on the "older" target net
        next_state_batch = torch.cat(batch.next_state)
        # TODO: could potentially restrict to only valid actions here --> but maybe not worth it
        next_state_values = target_net(next_state_batch).max(1)[0].detach()
        # compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

