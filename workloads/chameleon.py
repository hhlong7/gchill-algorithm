"""
chameleon importer: the task graph starpu really built for a chameleon tiled factorization -> a workload
- chameleon (on starpu) runs the factorization; starpu's fxt trace, turned into tasks.rec by
  starpu_fxt_tool, lists every task: its codelet name, its tiles, and the tasks it waited for
- tasks.rec: records split by blank lines, one "Key: value" per line. real kernels have a Model line and
  are numbered in SubmitOrder; starpu's own control tasks (data acquire, sync) have none: skipped, a task
  that waited on one waits for what it waited for (job ids are not in order, so this follows the links)
- deps on tasks outside the trace (matrix generation before tracing started) are dropped
- every tile must be b x b (matrix size a multiple of the tile size), so each kernel's shape is b
run from the repo root: python -m workloads.chameleon (reads data/chameleon/*.rec)
"""

import glob
import os
import re
from workloads.format import Kernel, Workload, save

#codelet name without its precision letter -> (kernel type, shape keys that are all b)
KERNELS = {
    "potrf": ("potrf", ("b",)),
    "trsm": ("trsm", ("m", "n")),
    "syrk": ("syrk", ("n", "k")),
    "gemm": ("gemm", ("m", "n", "k")),
    "getrf_nopiv": ("getrf", ("b",)),
}
DTYPE_BYTES = {"s": 4} #fp32 traces only: the format has no fp64


def records(text):
    out = []
    for block in text.split("\n\n"):
        rec = dict(line.split(": ", 1) for line in block.strip().splitlines() if ": " in line)
        if "JobId" in rec:
            out.append(rec)
    return out


#one real task -> (kernel type, shape, precision letter)
def kernel_of(rec):
    name = rec["Name"]
    prec, base = name[0], name[1:]
    if prec not in DTYPE_BYTES or base not in KERNELS:
        raise NotImplementedError(f"job {rec['JobId']}: codelet {name} has no kernel mapping")
    tiles = {(int(r), int(c)) for r, c in re.findall(r"M(\d+)x(\d+)x\d+", rec["Parameters"])}
    if len(tiles) != 1 or len({*next(iter(tiles))}) != 1:
        raise NotImplementedError(f"job {rec['JobId']}: tiles {sorted(tiles)} are not all one square size")
    b = next(iter(tiles))[0]
    kind, keys = KERNELS[base]
    return kind, {key: b for key in keys}, prec


def load_rec(path, name):
    with open(path) as f:
        jobs = {r["JobId"]: r for r in records(f.read())}
    real = sorted((r for r in jobs.values() if "Model" in r), key=lambda r: int(r["SubmitOrder"]))
    kid = {r["JobId"]: i for i, r in enumerate(real)} #kernel ids in submission order

    #the real tasks a job waits for, looking through control tasks; unknown (ghost) jobs are dropped
    def waits_for(job, seen):
        for d in jobs[job].get("DependsOn", "").split():
            if d in kid:
                yield kid[d]
            elif d in jobs and d not in seen:
                seen.add(d)
                yield from waits_for(d, seen)

    kernels, precs = [], set()
    for r in real:
        kind, shape, prec = kernel_of(r)
        precs.add(prec)
        deps = tuple(sorted(set(waits_for(r["JobId"], set()))))
        kernels.append(Kernel(id=kid[r["JobId"]], type=kind, shape=shape, deps=deps))
    if len(precs) != 1:
        raise ValueError(f"{path}: mixed precisions {sorted(precs)}")
    return Workload(name=name, source="chameleon", dtype_bytes=DTYPE_BYTES[precs.pop()], kernels=tuple(kernels))


if __name__ == "__main__":
    for path in sorted(glob.glob("data/chameleon/*.rec")):
        name = os.path.splitext(os.path.basename(path))[0]
        w = load_rec(path, name)
        save(w, f"data/workloads/{name}.json")
        print(f"{w.name}: {len(w.kernels)} kernels -> data/workloads/{name}.json")