"""
mix builder: several workloads -> one workload, each part a separate request
- each part's kernel ids move past the previous parts' so they stay unique; its deps move with them
- every kernel of a part is released at the part's arrival time (plus its own arrival_ns, if it had one)
- parts never wait on each other: they only compete for the cores, the accelerator and dram
the mixes saved here all arrive at 0 (most contention). staggered arrivals get picked in the experiments,
once the kernel table gives real kernel times to space them by
run from the repo root: python -m workloads.mix
"""

from dataclasses import replace
from workloads.format import Workload, load, save, validate

#mix name -> the workloads in data/workloads it is made of, all arriving at 0
SUITE = {
    "mix_resnet50_bert_b1": ["resnet50_b1", "bert_base_s128_b1"],
    "mix_2resnet50_2bert_b1": ["resnet50_b1", "resnet50_b1", "bert_base_s128_b1", "bert_base_s128_b1"],
}


#parts: list of (workload, arrival_ns) -> one workload with every part's kernels
def mix(name, parts):
    if len({w.dtype_bytes for w, _ in parts}) != 1:
        raise ValueError("all parts need the same dtype_bytes")
    kernels = []
    for w, arrival_ns in parts:
        new_id = {k.id: len(kernels) + i for i, k in enumerate(w.kernels)}
        kernels += [replace(k, id=new_id[k.id], deps=tuple(new_id[d] for d in k.deps),
                            arrival_ns=arrival_ns + k.arrival_ns) for k in w.kernels]
    out = Workload(name=name, source="mix", dtype_bytes=parts[0][0].dtype_bytes, kernels=tuple(kernels))
    validate(out)
    return out


if __name__ == "__main__":
    for name, members in SUITE.items():
        w = mix(name, [(load(f"data/workloads/{m}.json"), 0.0) for m in members])
        save(w, f"data/workloads/{name}.json")
        print(f"{w.name}: {len(members)} requests, {len(w.kernels)} kernels -> data/workloads/{name}.json")