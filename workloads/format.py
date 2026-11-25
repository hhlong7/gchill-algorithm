"""
workload file format: one json file per dag, the same for every source (torch.fx, chameleon, stg)

{ "name": "cholesky_4x128", "source": "chameleon", "dtype_bytes": 4,
  "kernels": [ {"id": 0, "type": "potrf", "shape": {"b": 128}, "deps": [], "arrival_ns": 0.0},
               {"id": 1, "type": "trsm", "shape": {"m": 128, "n": 128}, "deps": [0], "arrival_ns": 0.0} ] }

only the dag: what each kernel is and what it waits for. how long a kernel takes comes from the kernel
table and when it ran comes from the simulator, so neither is stored here
"""

import json
from dataclasses import dataclass

#kernel type -> the shape keys it needs to look up its cost
#gemm: (m x k) @ (k x n), same letters as scale-sim. potrf: one b x b tile. trsm: m x m triangle,
#m x n right hand side. syrk: n x n out from n x k. elementwise and reduce: n elements
SHAPES = {
    "gemm": ("m", "n", "k"),
    "potrf": ("b",),
    "trsm": ("m", "n"),
    "syrk": ("n", "k"),
    "elementwise": ("n",),
    "reduce": ("n",),
}


@dataclass(frozen=True)
class Kernel:
    id: int
    type: str
    shape: dict
    deps: tuple = ()
    arrival_ns: float = 0.0


@dataclass(frozen=True)
class Workload:
    name: str
    source: str #where the dag came from: torchfx, chameleon, stg, test
    dtype_bytes: int #2 = bf16, 4 = fp32, one precision for the whole dag
    kernels: tuple


#kernel ids so every kernel comes after its deps. raises if there is a cycle
def topological(kernels):
    left = {k.id: set(k.deps) for k in kernels}
    order = []
    while left:
        ready = sorted(kid for kid, deps in left.items() if not deps)
        if not ready:
            raise ValueError(f"cycle among kernels {sorted(left)}")
        order += ready
        for kid in ready:
            del left[kid]
        for deps in left.values():
            deps.difference_update(ready)
    return order


#every rule a workload must follow, raises ValueError on the first one broken
def validate(w):
    if w.dtype_bytes not in (2, 4):
        raise ValueError(f"dtype_bytes {w.dtype_bytes} must be 2 (bf16) or 4 (fp32)")
    ids = [k.id for k in w.kernels]
    if len(set(ids)) != len(ids):
        raise ValueError("kernel ids must be unique")
    known = set(ids)
    for k in w.kernels:
        if k.type not in SHAPES:
            raise ValueError(f"kernel {k.id}: unknown type {k.type}, pick one of {list(SHAPES)}")
        if set(k.shape) != set(SHAPES[k.type]):
            raise ValueError(f"kernel {k.id}: {k.type} needs shape keys {SHAPES[k.type]}, got {tuple(k.shape)}")
        if any(not isinstance(v, int) or v <= 0 for v in k.shape.values()):
            raise ValueError(f"kernel {k.id}: shape values must be positive ints, got {k.shape}")
        if k.id in k.deps:
            raise ValueError(f"kernel {k.id} depends on itself")
        missing = set(k.deps) - known
        if missing:
            raise ValueError(f"kernel {k.id} depends on unknown kernels {sorted(missing)}")
        if k.arrival_ns < 0:
            raise ValueError(f"kernel {k.id}: arrival_ns {k.arrival_ns} must be >= 0")
    topological(w.kernels)


def load(path):
    with open(path) as f:
        raw = json.load(f)
    kernels = tuple(Kernel(id=k["id"], type=k["type"], shape=dict(k["shape"]), deps=tuple(k.get("deps", ())),
                           arrival_ns=float(k.get("arrival_ns", 0.0)))
                    for k in raw["kernels"])
    w = Workload(name=raw["name"], source=raw["source"], dtype_bytes=raw["dtype_bytes"], kernels=kernels)
    validate(w)
    return w


def save(w, path):
    validate(w)
    raw = {"name": w.name, "source": w.source, "dtype_bytes": w.dtype_bytes,
           "kernels": [{"id": k.id, "type": k.type, "shape": k.shape, "deps": list(k.deps),
                        "arrival_ns": k.arrival_ns} for k in w.kernels]}
    with open(path, "w") as f:
        json.dump(raw, f, indent=1)