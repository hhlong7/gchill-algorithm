"""
the kernel table (B2): (type, shape, resource, cores) -> (compute_ns, dram_bytes, source)
- one csv, data/kernel_table.csv, one row per key; shape is written as sorted k=v pairs ("k=256;m=256;n=256")
- resource is cpu (cores 1..4) or accel (cores 1)
- source is the tool plus a short hash of its config files, so rows from an old config can be found
- the runners (table.accel, table.cpu) add rows with add(); the simulator reads them with load() + lookup()

this is the file for the table, it would load and save the csv and look rows up
the keys are type, shape, resource, cores, and each row would have compute ns, dram bytes and sources
gem5 runner would write through the same add() func

"""

import csv
import glob
import hashlib
import os
from workloads.format import load as load_workload

PATH = "data/kernel_table.csv"
COLUMNS = ["type", "shape", "resource", "cores", "compute_ns", "dram_bytes", "source"]

def shape_key(shape):
    return ";".join(f"{k}={shape[k]}" for k in sorted(shape))

def config_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:8]


#everyt distinct type and shape in the saved workload would be sorted to list of (type, shape dict)
def distinct(pattern="data/workloads/*.json"):
    seen = {}
    for path in glob.glob(pattern):
        for k in load_workload(path).kernels:
            seen[(k.type, shape_key(k.shape))] = (k.type, dict(k.shape))
    return [seen[key] for key in sorted(seen)]

def load(path=PATH):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    table = {}
    for r in rows:
        r["cores"], r["compute_ns"], r["dram_bytes"] = int(r["cores"]), float(r["compute_ns"]), int(r["dram_bytes"])
        table[(r["type"], r["shape"], r["resource"], r["cores"])] = r
    return table

#add rows, replace any with the same key, and rewirte the file sorted
def add(rows, path=PATH):
    table = load(path)
    for r in rows:
        r = dict(r, shape=shape_key(r["shape"]))
        table[(r["type"], r["shape"], r["resource"], r["cores"])] = r
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=COLUMNS)
        out.writeheader()
        for key in sorted(table):
            out.writerow({c: table[key][c] for c in COLUMNS})

def lookup(table, kernel, resource, cores=1):
    r = table[(kernel.type, shape_key(kernel.shape), resource, cores)]
    return r["compute_ns"], r["dram_bytes"]