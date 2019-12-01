from python.rl_prefetcher import TableRLPrefetcher


class TableQLearningPrefetcher(TableRLPrefetcher):
    def __init__(self, state_vocab, action_vocab, epsilon=0.1, use_window=128, learning_rate=0.1, discount=0.9):
        super().__init__(state_vocab, action_vocab, epsilon=epsilon, use_window=use_window)

        self.learning_rate = learning_rate
        self.discount = discount

    def update_estimate(self, state, action, next_state, reward):
        state_idx = self.state_dict[state]
        action_idx = self.action_dict[action]

        # compute the next optimal action to be used for
        next_state_idx = self.state_dict[next_state]
        action_rewards = self.expected_rewards[next_state_idx].toarray()[0]
        valid_action_ids = self._get_valid_action_ids(action_rewards, next_state)
        next_action = self._get_address(action_rewards, valid_action_ids)

        next_action_idx = self.action_dict[next_action]

        old_reward_est = self.expected_rewards[state_idx, action_idx]
        learned_value = reward + self.discount * self.expected_rewards[next_state_idx, next_action_idx]
        self.expected_rewards[state_idx, action_idx] = \
            (1 - self.learning_rate) * old_reward_est + self.learning_rate * learned_value

    def update_reward_estimates(self, curr_address):
        # compute rewards for "stale" item (item that is outside of reward window)
        stale_item = self.choice_history_buffer.get_stale_item()
        if stale_item is not None:
            if stale_item.needs_reward:
                next_item = self.choice_history_buffer.get_next_pbi(stale_item)
                self.update_estimate(stale_item.state, stale_item.action, next_item.state, -1)


        # perform positive reward when correct item is prefetched
        causal_prefetch_item = self.choice_history_buffer.get_causal_prefetch_item(curr_address)
        if causal_prefetch_item is not None:
            delay = self.choice_history_buffer.step - causal_prefetch_item.step
            reward = (self.use_window - delay) / self.use_window
            next_item = self.choice_history_buffer.get_next_pbi(causal_prefetch_item)
            self.update_estimate(causal_prefetch_item.state, causal_prefetch_item.action, next_item.state, reward)
            causal_prefetch_item.reward()

        self.choice_history_buffer.remove_stale_item()
