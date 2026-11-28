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
from common.cores.arm.neoverse_v2 import L1D, L1I, L2, NeoverseV2, NeoverseV2_FP

#gem5 25.1's neoverse v2 has no unit for some arm op classes, so the first such instruction never issues and the
#run hangs: scalar fmadd (openblas potrf's dot), sve ptrue (sgemv), sve reductions, ... (openblas gemm uses
#none of them). add them to the fp / simd units, latencies close to arm's neoverse v2 software optimization guide
missing_ops = {"FloatMultAcc": 4, "SimdPredAlu": 1, "SimdReduceAdd": 4, "SimdReduceAlu": 4, "SimdReduceCmp": 4,
               "SimdFloatReduceAdd": 8, "SimdFloatReduceCmp": 4, "SimdDiv": 12, "SimdDotProd": 3, "SimdAes": 2,
               "SimdAesMix": 2, "SimdCrc": 2}
class FP_all_ops(NeoverseV2_FP):
    opList = [OpDesc(opClass=o.opClass, opLat=o.opLat, pipelined=o.pipelined) for o in NeoverseV2_FP.opList] \
             + [OpDesc(opClass=c, opLat=lat, pipelined=c != "SimdDiv") for c, lat in missing_ops.items()]

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
for cpu in system.cpu:
    for iq in cpu.instQueues:
        if isinstance(iq.fuPool.FUList[0], NeoverseV2_FP):
            iq.fuPool.FUList = [FP_all_ops(count=fu.count) for fu in iq.fuPool.FUList]
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