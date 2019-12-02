
class PrefetchBuffer:
    def __init__(self, use_window):
        self.buffer = []
        self.use_window = use_window
        self.step = 0

    def add_item(self, action, address, state):
        self.buffer.append(PrefetchBufferItem(action, address, state, self.step))
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
    def __init__(self, action, address, state, step):
        self.action = action
        self.address = address
        self.state = state
        self.step = step
        self.reward = None

