"""
how much data each kernel has to move at least (inputs read once, output written once) and how a whole
workload compares to the llc. only for sizing the suite: the real dram bytes come from gem5 / scale-sim
- gemm: a (m x k) + b (k x n) + c (m x n)
- potrf: reads and writes its b x b tile
- trsm: the m x m triangle + the m x n right hand side read and written
- syrk: a (n x k) + c (n x n) read and written
- elementwise: n in + n out (an add reads 2 inputs: an underestimate, fine for sizing)
- reduce: n in, the small output ignored
run from the repo root: python -m workloads.size
"""

import glob
from workloads.format import load

LLC_BYTES = 12 * 2**20


def min_bytes(kernel, dtype_bytes):
    s, t = kernel.shape, kernel.type
    if t == "gemm":
        elems = s["m"] * s["k"] + s["k"] * s["n"] + s["m"] * s["n"]
    elif t == "potrf":
        elems = 2 * s["b"] * s["b"]
    elif t == "trsm":
        elems = s["m"] * s["m"] + 2 * s["m"] * s["n"]
    elif t == "syrk":
        elems = s["n"] * s["k"] + 2 * s["n"] * s["n"]
    elif t == "elementwise":
        elems = 2 * s["n"]
    elif t == "reduce":
        elems = s["n"]
    elif t == "bgemm":
        elems = s["b"] * (s["m"] * s["k"] + s["k"] * s["n"] + s["m"] * s["n"])
    else:
        raise ValueError(f"no size rule for kernel type {t}")
    return elems * dtype_bytes


#total and largest kernel traffic, and how many kernels alone are bigger than the llc
def report(w):
    sizes = [min_bytes(k, w.dtype_bytes) for k in w.kernels]
    return {"name": w.name, "kernels": len(sizes), "total_mb": sum(sizes) / 2**20,
            "max_kernel_mb": max(sizes) / 2**20, "over_llc": sum(b > LLC_BYTES for b in sizes),
            "x_llc": sum(sizes) / LLC_BYTES}


if __name__ == "__main__":
    for path in sorted(glob.glob("data/workloads/*.json")):
        r = report(load(path))
        print(f"{r['name']:20s} {r['kernels']:4d} kernels  {r['total_mb']:8.1f} MB total ({r['x_llc']:5.1f}x llc)"
              f"  largest kernel {r['max_kernel_mb']:6.2f} MB  {r['over_llc']} over llc")