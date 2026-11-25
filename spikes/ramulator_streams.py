"""
spike A4.2: how much dram bandwidth is delivered when n kernels stream at once (ramulator 2.1)
- n streams, each reads 64 B lines in order from a random 4 KB page (where an allocator could put it),
  interleaved one line at a time => all n compete in the same controller queues
- 2 reads : 1 write (a gemm reads a and b, writes c)
- ddr4-2400, 1 channel, 1 rank => peak 19.2 GB/s. MOP4CLXOR mapping (xor bank hashing): one stream
  alone fills the banks, more streams cause row conflicts
- the frontend sends a request every cycle it can => dram is saturated, we measure what it delivers
fraction(n) = delivered(n) / peak, averaged over 3 placements (seeds)
run from the repo root: python spikes/ramulator_streams.py
"""

import os
import random
import ramulator

line = 64 #bytes per request
peak_gbps = 19.2 #ddr4-2400 on a 64 bit bus
mem_bytes = 1 << 33 #8 GB, where the streams can start
out = "results/ramulator"


#n streams from random pages, one line from each in turn; every 3rd line of a stream is a write
def write_trace(path, n, total, seed, writes_every=3):
    rng = random.Random(seed)
    bases = [rng.randrange(mem_bytes // 4096) * 4096 for _ in range(n)]
    with open(path, "w") as f:
        for i in range(total // n):
            op = "ST" if i % writes_every == writes_every - 1 else "LD"
            for base in bases:
                f.write(f"{op} {(base + i * line) % mem_bytes}\n")


def delivered_gbps(n, seed, total=32768):
    path = os.path.join(out, f"streams_{n}_{seed}.trace")
    write_trace(path, n, total, seed)
    dram = ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R", rank=1)
    ctrl = ramulator.controller.GenericDDR(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.MOP4CLXOR(),
    )
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl], channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    sim = ramulator.Simulation(ramulator.frontend.LoadStoreTrace(clock_ratio=1, path=path), mem)
    sim.run()
    return sim.stats["memory_system"]["controller"]["total_throughput_MBps"] / 1000


if __name__ == "__main__":
    os.makedirs(out, exist_ok=True)
    for n in [1, 2, 4, 8, 16, 32]:
        gbps = sum(delivered_gbps(n, seed) for seed in range(3)) / 3
        print(f"n={n:2d}  {gbps:5.2f} GB/s  fraction {gbps / peak_gbps:.2f}")