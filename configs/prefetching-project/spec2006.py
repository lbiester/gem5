# import the m5 (gem5) library created when gem5 is built
import m5
import os
import requests
# import all of the SimObjects
from m5.objects import *

# Add the common scripts to our path
m5.util.addToPath('../../')

# import the caches which we made
from caches import *

# import the SimpleOpts module
from common import SimpleOpts

# set default args
cpu_2006_base_dir = '/speccpu2006-clean/benchspec/CPU2006/'
#default_max_insts = 1000000000 # 1 billion
default_max_insts = 100000000 # 100 million

# Set the usage message to display
SimpleOpts.add_option('--maxinsts',
        help="Max instructions to run. Default: %s" % default_max_insts)
SimpleOpts.add_option("--rl_prefetcher",
                      help="Which RL prefetcher to use")
SimpleOpts.add_option('--reward_type',
                      help="Type of rewards to use with the RL prefetcher")

SimpleOpts.set_usage("usage: %prog [--maxinsts number] [--rl_prefetcher string] [--reward_type string] spec_program")

# Finalize the arguments and grab the opts so we can pass it on to our objects
(opts, args) = SimpleOpts.parse_args()

# Check if there was a binary passed in via the command line and error if
# there are too many arguments
if len(args) == 1:
    spec_program = args[0]
else:
    SimpleOpts.print_help()
    m5.fatal("Expected a spec program to execute as positional argument")

# Clear python server and set proper RL prefetcher to be used
# to make this work run apt-get install python-requests
if opts.rl_prefetcher is not None and opts.rl_prefetcher not in ["table_bandits", "table_q"]:
    raise Exception("Unsupported RL prefetcher")
elif opts.reward_type is None and opts.rl_prefetcher is not None:
    raise Exception("Must specify a reward type when using a RL prefetcher")
elif opts.reward_type not in [None, "positive_negative_basic", "linear", "curved", "retrospective_cache"]:
    raise Exception("Unsupported reward type")
else:
    requests.post("http://localhost:8080", data={"rl_prefetcher": opts.rl_prefetcher, "spec_program": spec_program,
                                                 "reward_type": opts.reward_type})


# TODO: add more than just the first input file?
if spec_program == "bzip2" or spec_program == "401":
    binary = [os.path.join(cpu_2006_base_dir,
        '401.bzip2/exe/bzip2_base.docker')]
    input_file = [os.path.join(cpu_2006_base_dir,
        '401.bzip2/run/run_base_test_docker.0000/input.program'), '5']
elif spec_program == "lbm":
    binary = [os.path.join(cpu_2006_base_dir,
        '470.lbm/exe/lbm_base.docker')]
    input_file = ['20', os.path.join(cpu_2006_base_dir,
        '470.lbm/run/run_base_test_docker.0000/reference.dat'), '0', '1',
        os.path.join(cpu_2006_base_dir,
            '470.lbm/run/run_base_test_docker.0000/100_100_130_cf_a.of')]
elif spec_program == "gobmk":
    binary = [os.path.join(cpu_2006_base_dir, 
        '445.gobmk/exe/gobmk_base.docker')]
    input_file = ['--quiet', '--mode', 'gtp', '<', os.path.join(cpu_2006_base_dir, 
        '445.gobmk/run/run_base_test_docker.0000/capture.tst')]
elif spec_program == "hmmer":
    binary = [os.path.join(cpu_2006_base_dir,
        '456.hmmer/exe/hmmer_base.docker')]
    input_file = ['--fixed', '0', '--mean', '325', '--num', '45000', '--sd', '200', '--seed', '0', os.path.join(cpu_2006_base_dir,
        '456.hmmer/run/run_base_test_docker.0000/bombesin.hmm')]
elif spec_program == "milc":
    binary = [os.path.join(cpu_2006_base_dir,
        '433.milc/exe/milc_base.docker')]
    input_file = ['<', os.path.join(cpu_2006_base_dir, 
        '433.milc/run/run_base_test_docker.0000/su3imp.in')]
elif spec_program == "namd":
    binary = [os.path.join(cpu_2006_base_dir,
        '444.namd/exe/namd_base.docker')]
    input_file = ['--input', os.path.join(cpu_2006_base_dir, 
        '444.namd/run/run_base_test_docker.0000/namd.input'), '--iterations', '1', '--output', 
        os.path.join(cpu_2006_base_dir, '444.namd/run/run_base_test_docker.0000/namd.out')]
elif spec_program == "omnetpp":
    binary = [os.path.join(cpu_2006_base_dir,
        '471.omnetpp/exe/omnetpp_base.docker')]
    input_file = [os.path.join(cpu_2006_base_dir, '471.omnetpp/run/run_base_test_docker.0000/omnetpp.ini')]
elif spec_program == "sphinx": # TODO: not yet running :(
    binary = [os.path.join(cpu_2006_base_dir,
        '482.sphinx3/exe/sphinx_livepretend_base.docker')]
    sphinx_base_dir = os.path.join(cpu_2006_base_dir,
        '482.sphinx3/run/run_base_test_docker.0000')
    input_file = [os.path.join(sphinx_base_dir, 'ctlfile'),
        sphinx_base_dir + '/',
        os.path.join(sphinx_base_dir, 'args.an4')]
elif spec_program == "astar": # TODO: not yet running :(
    binary = [os.path.join(cpu_2006_base_dir,
        '473.astar/exe/astar_base.docker')]
    input_file = [os.path.join(cpu_2006_base_dir,
        '473.astar/run/run_base_test_docker.0000/lake.cfg')]
elif spec_program == "libquantum":
    binary = [os.path.join(cpu_2006_base_dir,
        '462.libquantum/exe/libquantum_base.docker')]
    input_file = ['33', '5']
elif spec_program == "mcf":
    binary = [os.path.join(cpu_2006_base_dir, '429.mcf/exe/mcf_base.docker')]
    input_file = [os.path.join(cpu_2006_base_dir,
        '429.mcf/run/run_base_test_docker.0000/inp.in')]
elif spec_program == 'sjeng':
    binary = [os.path.join(cpu_2006_base_dir, '458.sjeng','exe','sjeng_base.amd64-m64-gcc41-nn')]
    input_file = [os.path.join(cpu_2006_base_dir,'458.sjeng','run','run_base_test_amd64-m64-gcc41-nn.0000','test.txt')]
elif spec_program == 'povray':
    binary = [os.path.join(cpu_2006_base_dir, '453.povray','exe','povray_base.amd64-m64-gcc41-nn')]
    input_file = [os.path.join(cpu_2006_base_dir,'453.povray','run','run_base_test_amd64-m64-gcc41-nn.0000','SPEC-benchmark-test.pov')]
    input_file += [os.path.join(cpu_2006_base_dir,'453.povray','run','run_base_test_amd64-m64-gcc41-nn.0000','SPEC-benchmark-test.ini')]
elif spec_program == 'h264ref':
    binary = [os.path.join(cpu_2006_base_dir, '464.h264ref','exe','h264ref_base.amd64-m64-gcc41-nn')]
    #input_file = ['-f ' + os.path.join(cpu_2006_base_dir, '464.h264ref','run','run_base_test_amd64-m64-gcc41-nn.0000','foreman_test_encoder_baseline.cfg')]
    input_file = []
elif spec_program =='soplex':
    binary = [os.path.join(cpu_2006_base_dir, '450.soplex', 'exe', 'soplex_base.amd64-m64-gcc41-nn')]
    #input_file = ['-m10000']
    #input_file = [os.path.join(cpu_2006_base_dir,'450.soplex','run','run_base_test_amd64-m64-gcc41-nn.0000','test.mps')]
    input_file = ['-m10000', os.path.join(cpu_2006_base_dir,'450.soplex','run','run_base_test_amd64-m64-gcc41-nn.0000','test.mps')]
elif spec_program == 'dealII':
    binary = [os.path.join(cpu_2006_base_dir,'447.dealII','exe','dealII_base.amd64-m64-gcc41-nn')]
    input_file = ['8']

else:
    m5.fatal('Given spec program is not supported')

output_file = 'spec_run.' + spec_program + '.out'

# create the system we are going to simulate
system = System()

# Set the clock fequency of the system (and all of its children)
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1.5GHz'
system.clk_domain.voltage_domain = VoltageDomain()

# Set up the system
system.mem_mode = 'timing'               # Use timing accesses
system.mem_ranges = [AddrRange('4GB')] # Create an address range

# Create a simple CPU
#system.cpu = TimingSimpleCPU()
system.cpu = MinorCPU()


# Create an L1 instruction and data cache
system.cpu.icache = L1ICache(opts)
system.cpu.dcache = L1DCache(opts)

# Connect the instruction and data caches to the CPU
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

# Create a memory bus, a coherent crossbar, in this case
system.l2bus = L2XBar()

# Hook the CPU ports up to the l2bus
system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)

# Create an L2 cache and connect it to the l2bus
system.l2cache = L2Cache(opts)
system.l2cache.connectCPUSideBus(system.l2bus)

# Use RL Naive Prefetcher
if opts.rl_prefetcher is not None:
    system.l2cache.prefetcher = RLNaivePrefetcher()
else:
    print("Warning: no rl_prefetcher specified, no prefetcher being used!")

# Create a memory bus
system.membus = SystemXBar()

# Connect the L2 cache to the membus
system.l2cache.connectMemSideBus(system.membus)

# create the interrupt controller for the CPU
system.cpu.createInterruptController()

# For x86 only, make sure the interrupts are connected to the memory
# Note: these are directly connected to the memory bus and are not cached
if m5.defines.buildEnv['TARGET_ISA'] == "x86":
    system.cpu.interrupts[0].pio = system.membus.master
    system.cpu.interrupts[0].int_master = system.membus.slave
    system.cpu.interrupts[0].int_slave = system.membus.master

# Connect the system up to the membus
system.system_port = system.membus.slave

# Create a DDR3 memory controller
system.mem_ctrl = DDR3_1600_8x8()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.master




# Create a process for a simple "Hello World" application
process = Process()
system.cpu.process = process
system.cpu.max_insts_any_thread = opts.maxinsts or default_max_insts

# Set the command
# cmd is a list which begins with the executable (like argv)
process.cmd = binary + input_file
process.output = output_file
# Set the cpu to use the process as its workload and create thread contexts
system.cpu.workload = process
system.cpu.createThreads()

# set up the root SimObject and start the simulation
root = Root(full_system = False, system = system)
# instantiate all of the objects we've created above
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()
print('Exiting @ tick %i because %s', m5.curTick(), exit_event.getCause())



