def positive_negative_basic_reward_function(causal_prefetch_item):
    """
    Consistent positive reward based on something being used within 128 steps (1, -1) rewards
    """
    if causal_prefetch_item is None:
        return -1
    return 1


def linear_reward_function(choice_history_buffer, causal_prefetch_item, use_window):
    """
    Linear reward function, positive for item that was used, negative one if not used
    """
    if causal_prefetch_item is None:
        return -1
    prefetch_delay = choice_history_buffer.step - causal_prefetch_item.step
    return (use_window - prefetch_delay) / use_window


def curved_reward_function(choice_history_buffer, causal_prefetch_item, use_window):
    raise NotImplementedError()

def retrospective_cache_reward_function(choice_history_buffer, causal_prefetch_item, use_window):
    raise NotImplementedError()


def compute_reward(reward_type, choice_history_buffer, causal_prefetch_item, use_window):
    """
    Return a reward using the proper reward function
    :param reward_type: One of our reward functions
    :param choice_history_buffer: the buffer of choices made by the model
    :param causal_prefetch_item: the item from which our address was prefetched (None if this is caused by a stale item)
    :param use_window: the window size for our buffer
    :return: the reward
    """
    if reward_type == "positive_negative_basic":
        return positive_negative_basic_reward_function(causal_prefetch_item)
    elif reward_type == "linear":
        return linear_reward_function(choice_history_buffer, causal_prefetch_item, use_window)
    elif reward_type == "curved":
        return curved_reward_function(choice_history_buffer, causal_prefetch_item, use_window)
    elif reward_type == "retrospective_cache":
        # TODO: we may need more params for the "retrospective" reward
        return retrospective_cache_reward_function(choice_history_buffer, causal_prefetch_item, use_window)