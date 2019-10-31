import os

def get_benchmark_to_filename():
    benchmark_to_filename = {}
    for filename in os.listdir("spec_stats"):
        benchmark_to_filename[filename.split(".")[1]] = os.path.join("spec_stats", filename)
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


def main():
    btf = get_benchmark_to_filename()
    ipc = get_stat_for_benchmark("system.cpu.ipc", btf)
    l1_miss_temp = get_stat_for_benchmark("system.cpu.dcache.overall_misses::total", btf)
    l2_miss_temp = get_stat_for_benchmark("system.l2cache.overall_misses::.cpu.data", btf)
    sim_insts = get_stat_for_benchmark("sim_insts", btf)

    l1_miss = {1000 * l1_miss_temp[k] / sim_insts[k] for k in l1_miss_temp}
    l2_miss = {1000 * l2_miss_temp[k] / sim_insts[k] for k in l2_miss_temp}


if __name__ == "__main__":
    main()
