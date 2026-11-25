import pytest
from workloads.format import Kernel, Workload, load
from workloads.mix import SUITE, mix


#a -> b, every kernel released at 0
def chain(dtype_bytes=4):
    ks = (Kernel(0, "gemm", {"m": 8, "n": 8, "k": 8}),
          Kernel(1, "elementwise", {"n": 64}, deps=(0,)))
    return Workload("chain", "test", dtype_bytes, ks)


#two copies of the chain: the second one's ids and deps move past the first, its kernels arrive at 100
def test_ids_deps_and_arrivals_shift():
    w = mix("m", [(chain(), 0.0), (chain(), 100.0)])
    got = [(k.id, k.deps, k.arrival_ns) for k in w.kernels]
    assert got == [(0, (), 0.0), (1, (0,), 0.0), (2, (), 100.0), (3, (2,), 100.0)]
    assert (w.name, w.source, w.dtype_bytes) == ("m", "mix", 4)


#ids that are not 0..n-1 and a kernel with its own arrival time
def test_sparse_ids_and_own_arrival():
    part = Workload("p", "test", 4, (Kernel(7, "reduce", {"n": 4}, arrival_ns=5.0),
                                     Kernel(3, "reduce", {"n": 4}, deps=(7,))))
    w = mix("m", [(chain(), 0.0), (part, 10.0)])
    assert [(k.id, k.deps, k.arrival_ns) for k in w.kernels][2:] == [(2, (), 15.0), (3, (2,), 10.0)]


def test_rejects_mixed_dtypes():
    with pytest.raises(ValueError, match="dtype_bytes"):
        mix("m", [(chain(4), 0.0), (chain(2), 0.0)])


#every saved suite mix loads and holds exactly its members' kernels, in order
def test_suite_files():
    for name, members in SUITE.items():
        parts = [load(f"data/workloads/{m}.json") for m in members]
        w = load(f"data/workloads/{name}.json")
        assert [(k.type, k.shape) for k in w.kernels] == [(k.type, k.shape) for p in parts for k in p.kernels]