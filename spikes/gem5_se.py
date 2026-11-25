"""
gem5 se-mode run of one kernel binary on neoverse v2 cores (gem5's configs/common/cores/arm/neoverse_v2.py)
- every core: its own l1i, l1d and l2 (the neoverse v2 classes)
- --mem ideal: l2s straight to a zero-latency, unlimited-bandwidth memory -> compute time only
- --mem ddr: l2s -> a shared 12 MiB llc -> one ddr4-2400 channel -> dram bytes
run by spikes/gem5_gemm.py inside the gchill-gem5 image, with tools/gem5 mounted at /gem5
"""

import argparse
import sys
import m5
from m5.objects import *

sys.path.append("/gem5/configs")
from common.cores.arm.neoverse_v2 import L1D, L1I, L2, NeoverseV2

parser = argparse.ArgumentParser()
parser.add_argument("--cores", type=int, default=1)
parser.add_argument("--mem", choices=["ideal", "ddr"], default="ideal")
parser.add_argument("--clock", default="3GHz")
parser.add_argument("cmd", nargs="+")
args = parser.parse_args()

system = System()
system.clk_domain = SrcClockDomain(clock=args.clock, voltage_domain=VoltageDomain())
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("8GiB")]

system.cpu = [NeoverseV2(cpu_id=i) for i in range(args.cores)]
system.membus = SystemXBar()
system.system_port = system.membus.cpu_side_ports

if args.mem == "ddr":
    system.llc_bus = L2XBar()
    system.llc = Cache(size="12MiB", assoc=12, tag_latency=20, data_latency=20, response_latency=20,
                       mshrs=128, tgts_per_mshr=16, write_buffers=32)
    system.llc.cpu_side = system.llc_bus.mem_side_ports
    system.llc.mem_side = system.membus.cpu_side_ports
    below_l2 = system.llc_bus.cpu_side_ports
    system.mem_ctrl = MemCtrl(dram=DDR4_2400_8x8(range=system.mem_ranges[0]))
else:
    below_l2 = system.membus.cpu_side_ports
    system.mem_ctrl = SimpleMemory(range=system.mem_ranges[0], latency="0ns", bandwidth="0GiB/s")
system.mem_ctrl.port = system.membus.mem_side_ports

process = Process(cmd=args.cmd, env=[f"OPENBLAS_NUM_THREADS={args.cores}"])
for cpu in system.cpu:
    cpu.addTwoLevelCacheHierarchy(L1I(), L1D(), L2())
    cpu.createInterruptController()
    cpu.connectAllPorts(below_l2, system.membus.cpu_side_ports, system.membus.mem_side_ports)
    cpu.workload = process
    cpu.createThreads()
system.workload = SEWorkload.init_compatible(args.cmd[0])

root = Root(full_system=False, system=system)
m5.instantiate()
event = m5.simulate()
print(f"exit: {event.getCause()} at tick {m5.curTick()}")