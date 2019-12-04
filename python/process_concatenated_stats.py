import sys


def main():
    filename = sys.argv[1]
    desired_instructions = int(sys.argv[2])
    with open(filename, "r") as f:
        found_instructions = False
        for line in f.readlines():
            if "sim_insts" in line and not found_instructions:
                sim_insts = int(line.split()[1])
                if sim_insts >= desired_instructions:
                    found_instructions = True
                    print("sim_insts:", sim_insts)
            elif "sim_insts" in line and found_instructions:
                break
            if found_instructions and "system.cpu.ipc" in line:
                print("ipc:", line.split()[1])
            if found_instructions and "system.cpu.dcache.overall_misses::total" in line:
                print("dcache:", line.split()[1])
            if found_instructions and "system.l2cache.overall_misses::.cpu.data" in line:
                print("l2cache:", line.split()[1])

if __name__ == "__main__":
    main()