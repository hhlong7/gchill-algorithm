"""
spike A3: one gemm on scale-sim => the accelerator's compute time and dram bytes
- array: 32 x 32 at 1 GHz, output stationary, bf16 (configs/scalesim/accel.cfg)
- scale-sim counts 1 byte words (hard-coded), so dram bytes = words read + written x dtype_bytes
- InterfaceBandwidth CALC => no memory stalls, Total Cycles is pure compute (simgrid adds the dram time)
checks against the roofline: cycles >= m n k / (rows x cols), bytes >= (m k + k n + m n) x dtype_bytes
run from the repo root: python spikes/scalesim_gemm.py
"""

import os
import pandas as pd
from scalesim.scale_sim import scalesim

cfg = "configs/scalesim/accel.cfg"
layout = "configs/scalesim/no_layout.csv" #scale-sim always opens one, even with no custom layout
run_name = "accel_32x32_os" #same as run_name in the cfg, the reports go in a folder with this name
clock_hz = 1e9
dtype_bytes = 2 #bf16
rows = cols = 32


#one m x k by k x n gemm => cycles, compute ns, dram bytes
def gemm(m, n, k, out="results/scalesim"):
    os.makedirs(out, exist_ok=True)
    topo = os.path.join(out, f"gemm_{m}_{n}_{k}.csv")
    with open(topo, "w") as f:
        f.write("Layer, M, N, K,\n") #every row ends in a comma: scale-sim drops the last field
        f.write(f"gemm, {m}, {n}, {k},\n")
    sim = scalesim(save_disk_space=True, verbose=False, config=cfg, topology=topo, layout=layout,
                   input_type_gemm=True)
    sim.run_scale(top_path=out)
    report = os.path.join(out, run_name)
    c = pd.read_csv(os.path.join(report, "COMPUTE_REPORT.csv"), skipinitialspace=True).iloc[0]
    d = pd.read_csv(os.path.join(report, "DETAILED_ACCESS_REPORT.csv"), skipinitialspace=True).iloc[0]
    words = d["DRAM IFMAP Reads"] + d["DRAM Filter Reads"] + d["DRAM OFMAP Writes"]
    return {"cycles": int(c["Total Cycles"]), "stalls": int(c["Stall Cycles"]),
            "compute_ns": c["Total Cycles"] / clock_hz * 1e9, "dram_bytes": int(words * dtype_bytes)}


if __name__ == "__main__":
    m = n = k = 256
    r = gemm(m, n, k)
    ideal = m * n * k / (rows * cols)
    least = (m * k + k * n + m * n) * dtype_bytes
    print(r)
    print(f"cycles / ideal     = {r['cycles'] / ideal:.2f}  (>= 1)")
    print(f"dram bytes / least = {r['dram_bytes'] / least:.2f}  (>= 1)")
    assert r["stalls"] == 0 and r["cycles"] >= ideal and r["dram_bytes"] >= least