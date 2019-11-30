import random

from python.qlearning import QLearningPrefetcher


def main():
    n_addressses = 500
    state_vocab = []
    addresses = list(range(n_addressses))
    pcs = list(range(0, n_addressses, 5))
    for address in addresses:
        for pc in pcs:
            state_vocab.append((address, pc))
    # this is a silly way to do this but works
    action_vocab = range(-500, 501)

    prefetcher = QLearningPrefetcher(state_vocab, action_vocab)
    while True:
        curr_state = random.choice(state_vocab)
        prefetcher.select_action(curr_state)


if __name__ == '__main__':
    main()