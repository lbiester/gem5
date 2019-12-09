# import the m5 (gem5) library created when gem5 is built
import datetime
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
default_max_insts = 100000000 # 100 million

# Set the usage message to display
SimpleOpts.add_option('--maxinsts',
        help='Max instructions to run. Default: %s' % default_max_insts)
SimpleOpts.add_option('--rl_prefetcher',
                      help='Which RL prefetcher to use')
SimpleOpts.add_option('--reward_type',
                      help='Type of rewards to use with the RL prefetcher')

SimpleOpts.set_usage('usage: %prog [--maxinsts number] [--rl_prefetcher string] [--reward_type string] spec_program')

# Finalize the arguments and grab the opts so we can pass it on to our objects
(opts, args) = SimpleOpts.parse_args()

# Check if there was a binary passed in via the command line and error if
# there are too many arguments
if len(args) == 1:
    spec_program = args[0]
else:
    SimpleOpts.print_help()
    m5.fatal('Expected a spec program to execute as positional argument')

# Clear python server and set proper RL prefetcher to be used
# to make this work run apt-get install python-requests
if opts.rl_prefetcher is not None and opts.rl_prefetcher not in ['table_bandits', 'table_q', 'DQN']:
    raise Exception('Unsupported RL prefetcher')
elif opts.reward_type is None and opts.rl_prefetcher is not None:
    raise Exception('Must specify a reward type when using a RL prefetcher')
elif opts.reward_type not in [None, 'positive_negative_basic', 'linear', 'curved', 'retrospective_cache']:
    raise Exception('Unsupported reward type')
else:
    requests.post('http://localhost:8080', data={'rl_prefetcher': opts.rl_prefetcher, 'spec_program': spec_program,
                                                 'reward_type': opts.reward_type})


if spec_program == 'bzip2':
    # NOTE: this is set up to be run within the gem5 directory
    binary = [os.path.join(cpu_2006_base_dir,
        '401.bzip2/exe/bzip2_base.docker')]
    input_file = [os.path.join(cpu_2006_base_dir,
        '401.bzip2/run/run_base_test_docker.0000/input.program'), '5']
elif spec_program == 'sjeng':
    # NOTE: this is set up to be run within the gem5 directory
    binary = [os.path.join(cpu_2006_base_dir, '458.sjeng','exe','sjeng_base.docker-nn')]
    input_file = [os.path.join(cpu_2006_base_dir,'458.sjeng','run','run_base_test_docker-nn.0000','test.txt')]
elif spec_program == 'povray':
    # NOTE: this needs to be run within the spec directory
    binary = [os.path.join(cpu_2006_base_dir, '453.povray','exe','povray_base.docker-nn')]
    input_file = [os.path.join(cpu_2006_base_dir,'453.povray','run','run_base_test_docker-nn.0000','SPEC-benchmark-test.pov')]
    input_file += [os.path.join(cpu_2006_base_dir,'453.povray','run','run_base_test_docker-nn.0000','SPEC-benchmark-test.ini')]
else:
    m5.fatal('Given spec program is not supported')

output_file = 'spec_run.' + spec_program + '.out'

# create the system we are going to simulate
system = System()

# Set the clock frequency of the system (and all of its children)
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
    print('Warning: no rl_prefetcher specified, no prefetcher being used!')

# Create a memory bus
system.membus = SystemXBar()

# Connect the L2 cache to the membus
system.l2cache.connectMemSideBus(system.membus)

# create the interrupt controller for the CPU
system.cpu.createInterruptController()

# For x86 only, make sure the interrupts are connected to the memory
# Note: these are directly connected to the memory bus and are not cached
if m5.defines.buildEnv['TARGET_ISA'] == 'x86':
    system.cpu.interrupts[0].pio = system.membus.master
    system.cpu.interrupts[0].int_master = system.membus.slave
    system.cpu.interrupts[0].int_slave = system.membus.master

# Connect the system up to the membus
system.system_port = system.membus.slave

# Create a DDR3 memory controller
system.mem_ctrl = DDR3_1600_8x8()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.master




# Create a process for a simple 'Hello World' application
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

# this code allows us to dump stats in intermediate states, every n instructions
dump_n_instructions = 100000
i = 0
while True:
    i += dump_n_instructions
    system.cpu.scheduleInstStop(0, dump_n_instructions, 'dump statistics')
    event = m5.simulate()
    if event.getCause() == 'dump statistics':
        m5.stats.dump()
        print(str(datetime.datetime.now()), 'Instruction Count: {}, Dumping stats'.format(i))
    else:
        print('Exiting @ tick %i because %s', m5.curTick(), event.getCause())
        break
