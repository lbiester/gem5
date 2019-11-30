import random
import sys

import numpy as np
from scipy.stats import bernoulli


class QLearningPrefetcher:
    def __init__(self, state_vocab, action_vocab, epsilon=0.1, use_window=128, learning_rate=0.1, discount=0.9,
                 table_based=True):
        if sys.version_info[0] != 3:
            raise Exception("Q Learning Prefetcher must be used with python 3!")

        # vocabs are list of possible state/action options
        # states are (address, pc) tuples, actions are addresses to prefetch
        self.state_vocab = state_vocab
        self.action_vocab = action_vocab
        self.state_dict = {state: i for i, state in enumerate(state_vocab)}
        self.action_dict = {action: i for i, action in enumerate(action_vocab)}
        self.epsilon = epsilon
        self.expected_rewards = np.zeros((len(self.state_vocab), len(self.action_vocab)))
        # store history of actions/addresses chosen and the state and timestep in which they were chosen
        self.choice_history_buffer = PrefetchBuffer(use_window)
        self.use_window = use_window

        self.learning_rate = learning_rate
        self.discount = discount

    def select_action(self, curr_state):
        self.update_reward_estimates(curr_state[0])

        state_idx = self.state_dict[curr_state]
        action_rewards = self.expected_rewards[state_idx]
        explore = bernoulli.rvs(self.epsilon)
        valid_action_ids = self._get_valid_action_ids(action_rewards, curr_state)
        if explore:
            action = random.choice(valid_action_ids)
            address = self.action_vocab[action]
        else:
            address = self._get_address(action_rewards, valid_action_ids)
        assert(address not in self.choice_history_buffer and address != curr_state[0])
        self.choice_history_buffer.add_item(address, curr_state)

        return address

    def _get_valid_action_ids(self, action_rewards, curr_state):
        return [i for i in range(len(action_rewards)) if self._is_valid_action(i, curr_state)]

    def _get_address(self, action_rewards, valid_action_ids):
        max_val = np.NINF
        max_addresses = []
        for i, reward_val in enumerate(action_rewards):
            address = self.action_vocab[i]
            if i in valid_action_ids:
                if reward_val > max_val:
                    max_addresses.clear()
                    max_addresses.append(address)
                    max_val = reward_val
                elif reward_val == max_val:
                    max_addresses.append(address)

        return random.choice(max_addresses)

    def _is_valid_action(self, action_index, curr_state):
        address = self.action_vocab[action_index]
        # want to make sure not to select the same address for pre-fetching twice before it is used
        # also want to make sure we are not pre-fetching the current address (it will be cached anyway)
        return address not in self.choice_history_buffer and address != curr_state[0]

    def update_estimate(self, state, action, next_state, next_action, reward):
        state_idx = self.state_dict[state]
        action_idx = self.action_dict[action]

        next_state_idx = self.state_dict[next_state]
        next_action_idx = self.action_dict[next_action]

        old_reward_est = self.expected_rewards[state_idx][action_idx]
        learned_value = reward + self.discount * self.expected_rewards[next_state_idx][next_action_idx]
        self.expected_rewards[state_idx][action_idx] = \
            (1 - self.learning_rate) * old_reward_est + self.learning_rate * learned_value

    def update_reward_estimates(self, curr_address):
        # compute rewards for "stale" item (item that is outside of reward window)
        stale_item = self.choice_history_buffer.get_stale_item()
        if stale_item is not None:
            if stale_item.needs_reward:
                next_item = self.choice_history_buffer.get_next_pbi(stale_item)
                self.update_estimate(stale_item.state, stale_item.address, next_item.state, next_item.address, -1)


        # perform positive reward when correct item is prefetched
        causal_prefetch_item = self.choice_history_buffer.get_causal_prefetch_item(curr_address)
        if causal_prefetch_item is not None:
            delay = self.choice_history_buffer.step - causal_prefetch_item.step
            reward = (self.use_window + 1 - delay) / self.use_window
            next_item = self.choice_history_buffer.get_next_pbi(causal_prefetch_item)
            self.update_estimate(causal_prefetch_item.state, causal_prefetch_item.address, next_item.state,
                                 next_item.address, reward)
            causal_prefetch_item.reward()

        self.choice_history_buffer.remove_stale_item()


class PrefetchBuffer:
    def __init__(self, use_window):
        self.buffer = []
        self.use_window = use_window
        self.step = 0

    def add_item(self, address, state):
        self.buffer.append(PrefetchBufferItem(address, state, self.step))
        self.step += 1

    def remove_stale_item(self):
        if len(self.buffer) > self.use_window:
            self.buffer.pop(0)

    def get_stale_item(self):
        if len(self.buffer) > self.use_window:
            return self.buffer[0]
        return None

    def get_causal_prefetch_item(self, address):
        # get the prefetch item in which an address was prefetched
        for pbi in self.buffer:
            if pbi.address == address:
                return pbi
        return None

    def get_next_pbi(self, pbi):
        return self.buffer[pbi.step - self.step + 1]

    def __contains__(self, address):
        return address in [pbi.address for pbi in self.buffer]


class PrefetchBufferItem:
    def __init__(self, address, state, step):
        self.address = address
        self.state = state
        self.step = step
        self.needs_reward = True

    def reward(self):
        self.needs_reward = False
