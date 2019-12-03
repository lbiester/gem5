import numpy as np 

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
    """
    Curved reward function, positive for item that was used within 20-50 steps, 
    up to negative 4 if used outside of that, and -5 if not used
    """

    if causal_prefetch_item is None:
        return -5

    # start with a normal distribution
    x = np.linspace(1,use_window,use_window)
    y = [1/(np.sqrt(2*np.pi*8**2)) * np.exp(-(i-36)**2/(2*8**2)) for i in x]
    # shift and scale (rewards <20 and > 50 are negative)
    y = [(i-0.01)*100 for i in y]
    # adjust rewards beyond 60 so that they're increasingly negative
    for i in range(36+8*3,use_window):
        y[i] = y[i-1]*1.02
    
    prefetch_delay = int(choice_history_buffer.step - causal_prefetch_item.step)
    
    return y[prefetch_delay]

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