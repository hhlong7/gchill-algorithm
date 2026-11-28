"""
accelerator rows of the kernel table from scale-sim (B2)
- array: configs/scalesim/accel.cfg (32 x 32, output stationary) at 1 GHz, fp32
- InterfaceBandwidth CALC => no memory stalls: cycles are pure compute, simgrid adds the dram time
- scale-sim counts 1 byte words, so dram bytes = (ifmap + filter reads + ofmap writes) x 4
- gemm: one scale-sim layer; bgemm: b gemms back to back => b x one gemm's cycles and bytes
- only gemm and bgemm get accel rows (open decision 3: the rest run on the cpu)
- all shapes go in one topology file (one layer each), so scale-sim runs once
checks: every gemm has cycles >= m n k / 1024 and bytes >= (m k + k n + m n) x 4
run from the repo root: python -m table.accel
"""
import os
import pandas as pd
from scalesim.scale_sim import scalesim
from table import table

CFG = "configs/scalesim/accel.cfg"
LAYOUT = "configs/scalesim/no_layout.csv" #scale-sim always opens one, even with no custom layout
RUN_NAME = "accel_32x32_os" #run_name in the cfg: the reports go in a folder with this name
OUT = "results/scalesim"
CLOCK_HZ = 1e9
DTYPE_BYTES = 4 #fp32
PES = 32 * 32

#[(m, n, k)] -> [(cycles, dram bytes)] in the same order, from one scale-sim run
def run_gemms(shapes, out=OUT):
    os.makedirs(out, exist_ok=True)
    topo = os.path.join(out, "table_gemms.csv")
    with open(topo, "w") as f:
        f.write("Layer, M, N, K,\n") #every row ends in a comma: scale-sim drops the last field
        for i, (m, n, k) in enumerate(shapes):
            f.write(f"g{i}, {m}, {n}, {k},\n")
    sim = scalesim(save_disk_space=True, verbose=False, config=CFG, topology=topo, layout=LAYOUT,
                   input_type_gemm=True)
    sim.run_scale(top_path=out)
    report = os.path.join(out, RUN_NAME)
    c = pd.read_csv(os.path.join(report, "COMPUTE_REPORT.csv"), skipinitialspace=True)
    d = pd.read_csv(os.path.join(report, "DETAILED_ACCESS_REPORT.csv"), skipinitialspace=True)
    assert len(c) == len(d) == len(shapes) and (c["Stall Cycles"] == 0).all()
    words = d["DRAM IFMAP Reads"] + d["DRAM Filter Reads"] + d["DRAM OFMAP Writes"]
    return [(int(cy), int(wd) * DTYPE_BYTES) for cy, wd in zip(c["Total Cycles"], words)]

def rows():
    kernels = [(t, s) for t, s in table.distinct() if t in ("gemm", "bgemm")]
    gemms = sorted({(s["m"], s["n"], s["k"]) for _, s in kernels})
    result = dict(zip(gemms, run_gemms(gemms)))
    source = "scalesim:" + table.config_hash(CFG)
    out = []
    for t, s in kernels:
        m, n, k = s["m"], s["n"], s["k"]
        cycles, nbytes = result[(m, n, k)]
        assert cycles >= m * n * k / PES and nbytes >= (m * k + k * n + m * n) * DTYPE_BYTES, (t, s)
        times = s.get("b", 1) #bgemm = b gemms
        out.append({"type": t, "shape": s, "resource": "accel", "cores": 1,
                    "compute_ns": times * cycles / CLOCK_HZ * 1e9, "dram_bytes": times * nbytes, "source": source})
    return out

if __name__ == "__main__":
    new = rows()
    table.add(new)
    for r in new:
        s = r["shape"]
        least = s.get("b", 1) * (s["m"] * s["k"] + s["k"] * s["n"] + s["m"] * s["n"]) * DTYPE_BYTES
        util = s.get("b", 1) * s["m"] * s["n"] * s["k"] / PES / r["compute_ns"]
        print(f"{r['type']:5s} {table.shape_key(s):22s} {r['compute_ns'] / 1e3:9.1f} us  util {util:4.2f}  "
              f"{r['dram_bytes'] / 2**20:7.2f} MiB  ({r['dram_bytes'] / least:4.2f}x the minimum)")
    print(f"{len(new)} accel rows -> {table.PATH}")