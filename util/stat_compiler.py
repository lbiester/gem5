import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_benchmark_to_filename(gem5_path):
    spec_stats_path = os.path.join(gem5_path,'spec_stats')
    benchmark_to_filename = {}
    for filename in os.listdir(spec_stats_path):
        benchmark_to_filename[filename.split(".")[1]] = os.path.join(spec_stats_path, filename)
    return benchmark_to_filename

def get_stat_for_benchmark(stat, btf):
    stat_for_benchmarks = {}
    for benchmark, benchmark_filename in btf.items():
        with open(benchmark_filename) as f:
            for line in f.readlines():
                split_line = line.split()
                if len(split_line) > 0 and split_line[0] == stat:
                    stat_for_benchmarks[benchmark] = float(split_line[1])
    return stat_for_benchmarks

def make_stat_hist(name, value_name, stat_for_benchmarks, gem5_path):
    names = list(stat_for_benchmarks.keys())
    values = list(stat_for_benchmarks.values())    
    g = sns.barplot(x=names,y=values,color='steelblue')
    g.set_xticklabels(labels=names,rotation=60)
    plt.tight_layout()
    plt.ylabel(value_name)
    plt.title('Histogram of {} for each Benchmark'.format(name))
    plt.savefig(os.path.join(gem5_path,'util/plots/{}.png'.format(name)), bbox_inches = 'tight')
    plt.clf()

def main():
    #gem5_path = 'c:/users/amrit/documents/eecs573/gem5/'
    gem5_path = '/gem5/'
    btf = get_benchmark_to_filename(gem5_path)
    ipc = get_stat_for_benchmark("system.cpu.ipc", btf)
    l1_miss_temp = get_stat_for_benchmark("system.cpu.dcache.overall_misses::total", btf)
    l2_miss_temp = get_stat_for_benchmark("system.l2cache.overall_misses::.cpu.data", btf)
    sim_insts = get_stat_for_benchmark("sim_insts", btf)
    l1_miss = {k: 1000 * l1_miss_temp[k] / sim_insts[k] for k in l1_miss_temp}
    l2_miss = {k: 1000 * l2_miss_temp[k] / sim_insts[k] for k in l2_miss_temp}
    make_stat_hist("IPC", "instructions per cycle", ipc, gem5_path)
    make_stat_hist("L1 CMPK", "misses per kilo-instruction", l1_miss, gem5_path)
    make_stat_hist("L2 CMPK", "misses per kilo-instruction", l2_miss, gem5_path)

if __name__ == "__main__":
    main()
