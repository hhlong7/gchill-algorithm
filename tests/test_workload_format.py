from dataclasses import replace
import pytest
from workloads.format import Kernel, Workload, load, save, validate, topological


#0 -> (1, 2) -> 3: a potrf, two trsms that need it, a gemm that needs both
def diamond():
    ks = (Kernel(0, "potrf", {"b": 64}),
          Kernel(1, "trsm", {"m": 64, "n": 64}, deps=(0,)),
          Kernel(2, "trsm", {"m": 64, "n": 64}, deps=(0,)),
          Kernel(3, "gemm", {"m": 64, "n": 64, "k": 64}, deps=(1, 2), arrival_ns=5.0))
    return Workload("diamond", "test", 4, ks)


#the diamond with one kernel changed
def broken(i, **change):
    w = diamond()
    ks = list(w.kernels)
    ks[i] = replace(ks[i], **change)
    return replace(w, kernels=tuple(ks))


#save then load gives back the same workload
def test_round_trip(tmp_path):
    w = diamond()
    path = tmp_path / "w.json"
    save(w, path)
    assert load(path) == w


def test_topological_puts_deps_first():
    order = topological(diamond().kernels)
    assert order.index(0) < order.index(1) < order.index(3)
    assert order.index(2) < order.index(3)


@pytest.mark.parametrize("w", [
    broken(0, deps=(3,)), #cycle 0 -> 1 -> 3 -> 0
    broken(1, deps=(9,)), #unknown dep
    broken(1, deps=(1,)), #depends on itself
    broken(3, type="conv"), #unknown type
    broken(3, shape={"m": 64, "n": 64}), #gemm without k
    broken(3, shape={"m": 64, "n": 64, "k": 0}), #not positive
    broken(2, id=1), #duplicate id
    broken(3, arrival_ns=-1.0), #arrives before time 0
    replace(diamond(), dtype_bytes=3), #not bf16 or fp32
])
def test_validate_rejects(w):
    with pytest.raises(ValueError):
        validate(w)