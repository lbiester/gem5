import pandas as pd 

def load_vocab(benchmark):

    delta_file = "vocab/{}.csv".format(benchmark)
    delta_df = pd.read_csv(delta_file)
    delta_df['pcs'] = delta_df['pc_deltas'].cumsum().astype('uint64')

    # only keep rows where address_diff is common (>=5 occurences)
    delta_df = delta_df.groupby('address_deltas').filter(lambda x : len(x)>=5)

    # states are address diffs, pc tuples
    # actions are address diffs
    address_diffs = delta_df['address_deltas'].to_list()
    pcs = delta_df['pcs'].to_list()
    return list(set(zip(address_diffs, pcs))), list(set(address_diffs))