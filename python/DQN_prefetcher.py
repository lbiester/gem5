import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import bernoulli
rom collections import namedtuple
from prefetch_buffer import PrefetchBuffer


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
    def __int__(self, input_dim, hidden_sizes, output_dim):
        # for now, assuming 2 hidden layers
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_sizes[0]),
                                nn.ReLU(),
                                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                nn.ReLU(),
                                nn.Linear(hidden_sizes[1], output_dim))

    def forward(self, x):
        x = self.fc(x)
        return x


class RNN(nn.Module)


class DQNPrefetcher:
    def __init__(self, state_vocab, action_vocab, epsilon=0.1, use_window=128):
        if sys.version_info[0] != 3:
            raise Exception("RL Prefetcher must be used with python 3!")

        # vocabs are list of possible state/action options
        # states are (address, pc) tuples, actions are deltas for addresses to prefetch
        self.state_vocab = state_vocab
        self.action_vocab = action_vocab
        self.state_dict = {state: i for i, state in enumerate(state_vocab)}
        self.action_dict = {action: i for i, action in enumerate(action_vocab)}
        self.epsilon = epsilon
        # store history of actions/addresses chosen and the state and timestep in which they were chosen
        self.choice_history_buffer = PrefetchBuffer(use_window)
        self.use_window = use_window
    
    def select_action(self, curr_state):
        self.update_reward_estimates(curr_state[0])

        state_idx = self.state_dict[curr_state]
        action_rewards = self.expected_rewards[state_idx].toarray()[0]
        explore = bernoulli.rvs(self.epsilon)
        valid_action_ids = self._get_valid_action_ids(action_rewards, curr_state)
        if explore:
            action = random.choice(valid_action_ids)
            address_diff = self.action_vocab[action]
        else:
            address_diff = self._get_address(action_rewards, valid_action_ids)
        address = curr_state[0] + address_diff
        # TODO: it seems like we should also check that the address when combined with the diff doesn't go off the edge
        # of the address space
        assert(address not in self.choice_history_buffer and address != curr_state[0])
        self.choice_history_buffer.add_item(address_diff, address, curr_state)

        return address

    def _get_valid_action_ids(self, action_rewards, curr_state):
        return [i for i in range(action_rewards.shape[0]) if self._is_valid_action(i, curr_state)]

    def _get_address(self, action_rewards, valid_action_ids):
        max_val = np.NINF
        max_addresses = []
        for i in valid_action_ids:
            address_diff = self.action_vocab[i]
            reward_val = action_rewards[i]
            if reward_val > max_val:
                max_addresses.clear()
                max_addresses.append(address_diff)
                max_val = reward_val
            elif reward_val == max_val:
                max_addresses.append(address_diff)
        return random.choice(max_addresses)

    def _is_valid_action(self, action_index, curr_state):
        address_diff = self.action_vocab[action_index]
        address = curr_state[0] + address_diff
        # want to make sure not to select the same address for pre-fetching twice before it is used
        # also want to make sure we are not pre-fetching the current address (it will be cached anyway)
        return address not in self.choice_history_buffer and address_diff != 0

    def update_reward_estimates(self, curr_address):
        raise NotImplementedError
