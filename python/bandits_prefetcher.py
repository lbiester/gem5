import random
import numpy as np
from scipy.stats import bernoulli
from rl_prefetcher import TableRLPrefetcher


# TODO: how best to assign rewards? Should "too soon" of use be penalized? Should max reward be > 1?
# what if something is used twice? Shouldn't this increase reward? for now no extra reward is given


class BanditsPrefetcher(TableRLPrefetcher):
    def __init__(self, state_vocab, action_vocab, epsilon=0.1, use_window=128):
        TableRLPrefetcher.__init__(state_vocab, action_vocab, epsilon, use_window)    

    def update_estimate(self, state, action, reward):
        state_idx = self.state_dict[state]
        action_idx = self.action_idx[action]
        old_reward_est = self.expected_rewards[state_idx][action_idx]
        choice_count = self.choice_counts[state_idx][action_idx]
        self.expected_rewards[state_idx][action_idx] = old_reward_est + (1 / float(choice_count)) * (reward - old_reward_est)

    def update_reward_estimates(self, curr_address, curr_step):
        # compute rewards for "stale" item (item outside of prefetch buffer window)
        stale_item = self.choice_history_buffer.get_stale_item()
        if stale_item is not None:
            if stale_item.needs_reward:
                self.update_estimate(state_item.state, stale_item.action, -1)

        # compute reward for past decision to prefetch current address (if it was prefetched before)
        causal_prefetch_item = self.choice_history_buffer.get_causal_prefetch_item(curr_address)
        if causal_prefetch_item is not None:
            delay = self.choice_history_buffer.step - causal_prefetch_item.step
            reward = (self.use_window - delay) / self.use_window
            self.update_estimate(causal_prefetch_item.state, causal_prefetch_item.action, reward)
            causal_prefetch_item.reward()

        self.choice_history_buffer.remove_stale_item()
