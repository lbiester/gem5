import pickle

import numpy as np

def load_vocab(address_file="vocab/sjeng_address.pkl", pc_file="vocab/sjeng_pc.pkl"):
    # TODO: consider loading other files?
    addresses = _read_address_file(address_file)
    pcs = _read_address_file(pc_file)
    # states are address, pc tuples
    return list(set(zip(addresses, pcs))), list(set(pcs))


def _read_address_file(filename):
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
        arr = loaded[0] + loaded[2:]
        return np.cumsum(arr)
