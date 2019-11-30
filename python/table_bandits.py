import random
import numpy as np
from scipy.stats import bernoulli


# TODO: how best to assign rewards? Should "too soon" of use be penalized? Should max reward be > 1?
# what if something is used twice? Shouldn't this increase reward? for now no extra reward is given

class ContextBandit:
    def __init__(self, state_vocab, action_vocab, epsilon=0.1, use_window=128):
        # vocabs are list of possible state/action options
        # states are (address, pc) tuples, actions are addresses to prefetch
        self.state_vocab = state_vocab
        self.action_vocab = action_vocab
        self.state_dict = {state: i for i, state in enumerate(state_vocab)}
        self.action_dict = {action: i for i, action in enumerate(action_vocab)}
        self.epsilon = epsilon
        self.expected_rewards = np.zeros((len(self.state_vocab), len(self.action_vocab)))
        # count of previous choices is used for updating reward estimates
        self.choice_counts = np.zeros((len(self.state_vocab), len(self.action_vocab)))
        # store history of actions/addresses chosen and the state and timestep in which they were chosen
        self.choice_history_buffer = []

    def select_action(self, curr_state, curr_step):
        state_idx = self.state_dict[curr_state]
        action_rewards = self.expected_rewards[state_idx]
        explore = bernoulli.rvs(self.epsilon)
        dup_action = True
        # want to make sure not to select the same address for pre-fetching twice before it is used
        # also want to make sure we are not pre-fetching the current address (it will be cached anyway)
        while dup_action:
            if explore:
                action = random.choice(np.arange(len(action_rewards)))
            else:
                action_opts = [i for i, val in enumerate(action_rewards) if val == max(action_rewards)]
                action = random.choice(action_opts)
            address = self.action_vocab[action]
            if address not in [x[0] for x in self.choice_history] and address != state[0]:
                dup_action = False
        self.choice_history_buffer.append((address, curr_state, curr_step))
        self.choice_counts[state_idx][action] += 1
        return address

    def update_estimate(self, state, action, reward):
        state_idx = self.state_dict[state]
        action_idx = self.action_idx[action]
        old_reward_est = self.expected_rewards[state_idx][action_idx]
        choice_count = self.choice_counts[state_idx][action_idx]
        self.expected_rewards[state_idx][action_idx] = old_reward_est + (1 / float(choice_count)) * (reward - old_reward_est)

    def update_reward_estimates(self, curr_address, curr_step):
        # check for, penalize, and remove choices that were selected more than <use_window> decisions ago
        old_choices = [x for x in self.choice_history_buffer if (curr_step - x[2]) > self.use_window]
        for choice in old_choices:
            choice_address, choice_state, choice_step = choice
            reward = -1
            self.update_estimate(choice_state, choice_address, reward)
            old_choices.remove(choice)
        # check if curr_address was selected for prefetch in last <use_window> decisions
        choices = [x for x in self.choice_history_buffer if x[0] == curr_address]
        if choices:
            assert len(choices) == 1
            choice = choices[0]
            choice_address, choice_state, choice_step = choice
            delay = curr_step - choice_step
            # calculate numerator for reward fraction --> bigger delay, smaller reward
            num = use_window + 1 - delay
            reward = num / use_window
            self.update_estimate(choice_state, choice_address, reward)
            choices.remove(choice)

