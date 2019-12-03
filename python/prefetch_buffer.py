
class PrefetchBuffer:
    def __init__(self, use_window):
        self.buffer = []
        self.use_window = use_window
        self.step = 0

    def add_item(self, action, address, state):
        self.buffer.append(PrefetchBufferItem(action, address, state, self.step))
        self.step += 1

    def add_null(self):
        self.buffer.append(None)
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
            if pbi is not None and pbi.address == address:
                return pbi
        return None

    def get_next_pbi(self, pbi):
        base_index = pbi.step - self.step + 1
        for i in range(base_index, len(self.buffer)):
            if self.buffer[i] is not None:
                return self.buffer[i]
        return None

    def __contains__(self, address):
        for pbi in self.buffer:
            if pbi is not None and pbi.address == address:
                return True
        return False


class PrefetchBufferItem:
    def __init__(self, action, address, state, step):
        self.action = action
        self.address = address
        self.state = state
        self.step = step
        self.reward = None

    def set_reward(self, reward):
        self.reward = reward
