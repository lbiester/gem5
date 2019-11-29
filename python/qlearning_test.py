import random

from python.qlearning import QLearningPrefetcher


def main():
    state_vocab = []
    addresses = list(range(1000))
    pcs = list(range(0, 10000, 5))
    for address in addresses:
        for pc in pcs:
            state_vocab.append((address, pc))

    prefetcher = QLearningPrefetcher(state_vocab, addresses)
    while True:
        curr_state = random.choice(state_vocab)
        prefetcher.select_action(curr_state)


if __name__ == '__main__':
    main()