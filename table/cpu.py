"""
cpu rows of the kernel table from gem5 (B2): openblas kernels (table/cpu_kernels.c) on neoverse v2 cores in se mode
- every kernel type in the suite gets cpu rows (gemm and bgemm too: a policy may put them on the cpu)
- compute_ns: ideal memory (zero latency, unlimited bandwidth) at 1..4 cores, so only compute is timed;
  inputs start warm (no cache scrub, see table/cpu_kernels.c).
  elementwise and reduce are single-thread loops: 1 core only
- dram_bytes: 1 core, 12 MiB llc + ddr4-2400, cold caches: what the kernel reads and writes + the dirty lines
  written back after it. the same bytes for every core count
- each run's gem5 output stays in results/gem5/table/<run>/; a run whose run.log says ok is not run again
- WORKERS gem5 runs at a time (one gem5 uses one cpu; the colima vm has 8)
checks: every run prints ok; time >= flops / (cores x 32 flop/cycle x 3 GHz); bytes >= workloads.size.min_bytes
run from the repo root: python -m table.cpu

"""
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from table import table
from workloads.size import min_bytes

IMAGE = "gchill-gem5"
SRC = "table/cpu_kernels.c"
SE = "spikes/gem5_se.py"
BIN = "results/gem5/cpu_kernels"
OUT = "results/gem5/table"
CORES = (1, 2, 3, 4)
SINGLE = {"elementwise", "reduce"} #plain loops on one thread
DIMS = {"gemm": "mnk", "bgemm": "bmnk", "potrf": "b", "getrf": "b", "trsm": "mn", "syrk": "nk",
        "elementwise": "n", "reduce": "n"} #argument order of cpu_kernels
WORKERS = 6
PEAK = 32 * 3e9 #flop/s per core: 4 x 128-bit fma pipes at 3 GHz

#run a command in the image with the repo at /repo (as you, so results/ stays yours) and gem5 at /gem5
def docker(*cmd):
    return subprocess.run(["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                           "-v", f"{os.getcwd()}:/repo", "-v", f"{os.path.realpath('tools/gem5')}:/gem5:ro",
                           "-w", "/repo", IMAGE, *cmd], check=True, capture_output=True, text=True)

def build():
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    docker("gcc", "-O2", "-static", "-I/opt/openblas/include", SRC, "-o", BIN, "/opt/openblas/lib/libopenblas.a",
           "-lgfortran", "-lpthread", "-lm")

#stats.txt -> one {stat name: value} per dump
def stat_blocks(path):
    blocks = []
    for line in open(path):
        if line.startswith("---------- Begin"):
            blocks.append({})
        elif blocks and line.strip() and not line.startswith("----------"):
            name, value = line.split()[:2]
            blocks[-1][name] = float(value) if value.replace(".", "", 1).isdigit() else value
    return blocks

def flops(t, s):
    return {"gemm": lambda: 2 * s["m"] * s["n"] * s["k"], "bgemm": lambda: 2 * s["b"] * s["m"] * s["n"] * s["k"],
            "potrf": lambda: s["b"] ** 3 / 3, "getrf": lambda: 2 * s["b"] ** 3 / 3, "trsm": lambda: s["m"] ** 2 * s["n"],
            "syrk": lambda: s["n"] ** 2 * s["k"], "elementwise": lambda: s["n"], "reduce": lambda: s["n"]}[t]()

#one gem5 run -> its stats blocks (1 = the kernel, 2 = the write-back after it in ddr runs)
def run(t, s, cores, mem):
    dims = [str(s[d]) for d in DIMS[t]]
    out = f"{OUT}/{t}_{'_'.join(dims)}_c{cores}_{mem}"
    log = f"{out}/run.log"
    if not (os.path.exists(log) and f"{t} ok" in open(log).read()):
        start = time.time()
        r = docker("/gem5/build/ARM/gem5.opt", "-q", "-d", out, SE, "--cores", str(cores), "--mem", mem,
                   "--", BIN, "1" if mem == "ddr" else "0", t, *dims)
        with open(log, "w") as f:
            f.write(r.stdout + r.stderr)
        assert f"{t} ok" in r.stdout, f"{out} did not finish, see {log}"
        print(f"  {out}: {time.time() - start:.0f} s", flush=True)
    return stat_blocks(f"{out}/stats.txt")

def rows():
    build()
    kernels = table.distinct()
    jobs = [(t, s, c, "ideal") for t, s in kernels for c in ((1,) if t in SINGLE else CORES)]
    jobs += [(t, s, 1, "ddr") for t, s in kernels]
    with ThreadPoolExecutor(WORKERS) as pool:
        blocks = list(pool.map(lambda job: run(*job), jobs))
    result = {(t, table.shape_key(s), c, mem): b for (t, s, c, mem), b in zip(jobs, blocks)}
    source = "gem5:" + table.config_hash(SRC, SE)
    dram = "system.mem_ctrl.dram."
    out = []
    for t, s in kernels:
        key = table.shape_key(s)
        kernel, writeback = result[(t, key, 1, "ddr")][:2]
        nbytes = int(kernel.get(dram + "bytesRead::total", 0) + kernel.get(dram + "bytesWritten::total", 0)
                     + writeback.get(dram + "bytesWritten::total", 0)) #gem5 leaves out a stat that stayed 0
        assert nbytes >= min_bytes(SimpleNamespace(type=t, shape=s), 4), (t, s, nbytes)
        for c in ((1,) if t in SINGLE else CORES):
            ns = result[(t, key, c, "ideal")][0]["simTicks"] / 1000
            assert ns >= flops(t, s) / (c * PEAK) * 1e9, (t, s, c, ns)
            out.append({"type": t, "shape": s, "resource": "cpu", "cores": c, "compute_ns": ns,
                        "dram_bytes": nbytes, "source": source})
    return out

if __name__ == "__main__":
    new = rows()
    table.add(new)
    by_kernel = {}
    for r in new:
        by_kernel.setdefault((r["type"], table.shape_key(r["shape"])), []).append(r)
    for (t, key), rs in by_kernel.items():
        us = "  ".join(f"{r['compute_ns'] / 1e3:8.1f}" for r in rs)
        least = min_bytes(SimpleNamespace(type=t, shape=rs[0]["shape"]), 4)
        print(f"{t:11s} {key:22s} us at 1..{len(rs)} cores: {us}  "
              f"{rs[0]['dram_bytes'] / 2**20:7.2f} MiB ({rs[0]['dram_bytes'] / least:4.2f}x the minimum)")
    print(f"{len(new)} cpu rows -> {table.PATH}")