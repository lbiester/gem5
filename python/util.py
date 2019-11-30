import pickle

import numpy as np

def load_vocab(address_file="vocab/sjeng_address.pkl", pc_file="vocab/sjeng_pc.pkl"):
    # TODO: consider loading other files?
    addresses, address_diffs = _read_address_file(address_file)
    pcs, _ = _read_address_file(pc_file)
    # states are address, pc tuples
    return list(set(zip(addresses, pcs))), list(set(address_diffs))


def _read_address_file(filename):
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
        arr = loaded[0] + loaded[2:]
        return np.cumsum(arr), arr[1:]
