import random

from python.table_q_prefetcher import TableQLearningPrefetcher


def main():
    n_addressses = 500
    state_vocab = []
    addresses = list(range(n_addressses))
    pcs = list(range(0, n_addressses, 5))
    # this is a silly way to do this but works
    action_vocab = range(-500, 501)

    for address_diff in action_vocab:
        for pc in pcs:
            if random.random() > 0.3:
                state_vocab.append((address_diff, pc))

    prefetcher = TableQLearningPrefetcher(state_vocab, action_vocab, "linear")
    while True:
        address = random.choice(addresses)
        pc = random.choice(pcs)
        prefetcher.select_action(address, pc)


if __name__ == '__main__':
    main()