import random
from IPython import embed
import numpy as np
from scipy.stats import bernoulli
from python.rl_prefetcher import TableRLPrefetcher
from python.reward_functions import compute_reward

# TODO: how best to assign rewards? Should "too soon" of use be penalized? Should max reward be > 1?
# what if something is used twice? Shouldn't this increase reward? for now no extra reward is given


class BanditsPrefetcher(TableRLPrefetcher):
    def __init__(self, state_vocab, action_vocab, reward_type, epsilon=0.1, use_window=128):
        super().__init__(state_vocab, action_vocab, reward_type, epsilon, use_window)

    def update_estimate(self, state, action, reward):
        state_idx = self.state_dict[state]
        action_idx = self.action_dict[action]
        old_reward_est = self.expected_rewards[state_idx][action_idx]
        choice_count = self.choice_counts[state_idx][action_idx]
        self.expected_rewards[state_idx][action_idx] = old_reward_est + (1 / float(choice_count)) * (reward - old_reward_est)

    def update_reward_estimates(self, curr_address):
        # compute rewards for "stale" item (item outside of prefetch buffer window)
        stale_item = self.choice_history_buffer.get_stale_item()
        if stale_item is not None:
            if stale_item.reward is None: 
                stale_item.reward = compute_reward(self.reward_type, self.choice_history_buffer, None, self.use_window)

            self.update_estimate(stale_item.state, stale_item.action, stale_item.reward)

        # compute reward for past decision to prefetch current address (if it was prefetched before)
        causal_prefetch_item = self.choice_history_buffer.get_causal_prefetch_item(curr_address)
        if causal_prefetch_item is not None:
            reward = compute_reward(self.reward_type, self.choice_history_buffer, causal_prefetch_item, self.use_window)

            #delay = self.choice_history_buffer.step - causal_prefetch_item.step
            #reward = (self.use_window - delay) / self.use_window
            self.update_estimate(causal_prefetch_item.state, causal_prefetch_item.action, reward)
            causal_prefetch_item.set_reward(reward)

        self.choice_history_buffer.remove_stale_item()
