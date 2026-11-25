"""
gem5 spike: one openblas sgemm on gem5's neoverse v2 model in se mode, run in the gchill-gem5 image
- ideal memory at 1, 2 and 4 cores: compute time only, and whether openblas threads scale in se mode
- caches + ddr4 at 1 core: time with real memory, and the dram bytes (gemm reads and writes + the dirty
  lines it left, written back by the scrub after it)
- also prints how long gem5 itself took, to size the kernel table runs
run from the repo root: .venv/bin/python spikes/gem5_gemm.py
"""

import os
import subprocess
import time

IMAGE = "gchill-gem5"
OUT = "results/gem5"
N = 256


#run a command in the image with the repo at /repo (as you, so results/ stays yours) and gem5 at /gem5
def docker(*cmd):
    repo = os.getcwd()
    subprocess.run(["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{repo}:/repo", "-v", f"{os.path.realpath('tools/gem5')}:/gem5:ro", "-w", "/repo",
                    IMAGE, *cmd], check=True, capture_output=True, text=True)


def build():
    os.makedirs(OUT, exist_ok=True)
    docker("gcc", "-O2", "-static", "-I/opt/openblas/include", "spikes/gem5_gemm.c", "-o", f"{OUT}/gemm",
           "/opt/openblas/lib/libopenblas.a", "-lgfortran", "-lpthread", "-lm")


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


#-> (simulated gemm ns, dram bytes, gem5 host seconds)
def run(cores, mem):
    out = f"{OUT}/n{N}_c{cores}_{mem}"
    start = time.time()
    docker("/gem5/build/ARM/gem5.opt", "-q", "-d", out, "spikes/gem5_se.py", "--cores", str(cores), "--mem", mem,
           "--", f"{OUT}/gemm", str(N), str(N), str(N))
    gemm, writeback = stat_blocks(f"{out}/stats.txt")[:2]
    dram = "system.mem_ctrl.dram." if mem == "ddr" else "system.mem_ctrl."
    moved = (gemm.get(dram + "bytesRead::total", 0) + gemm.get(dram + "bytesWritten::total", 0)
             + writeback.get(dram + "bytesWritten::total", 0))
    return gemm["simSeconds"] * 1e9, moved, time.time() - start


if __name__ == "__main__":
    build()
    flops, min_bytes = 2 * N**3, 4 * 4 * N * N #a, b, c read + c written, fp32
    print(f"sgemm {N}^3: {flops / 1e6:.1f} mflop, at least {min_bytes / 2**20:.2f} MiB of dram traffic")
    for cores, mem in [(1, "ideal"), (2, "ideal"), (4, "ideal"), (1, "ddr")]:
        ns, moved, host_s = run(cores, mem)
        line = f"cores={cores} mem={mem:5s}  {ns / 1e3:7.1f} us  {flops / ns:5.1f} gflop/s  gem5 took {host_s:4.0f} s"
        print(line + (f"  dram {moved / 2**20:.2f} MiB ({moved / min_bytes:.2f}x the minimum)" if mem == "ddr" else ""))