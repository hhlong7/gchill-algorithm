import pytest
from workloads.format import Kernel, Workload
from workloads.size import min_bytes, report, LLC_BYTES


@pytest.mark.parametrize("kernel, elems", [
    (Kernel(0, "gemm", {"m": 2, "n": 3, "k": 4}), 2 * 4 + 4 * 3 + 2 * 3),
    (Kernel(0, "potrf", {"b": 5}), 2 * 25),
    (Kernel(0, "trsm", {"m": 4, "n": 3}), 16 + 2 * 12),
    (Kernel(0, "syrk", {"n": 3, "k": 5}), 15 + 2 * 9),
    (Kernel(0, "elementwise", {"n": 10}), 20),
    (Kernel(0, "reduce", {"n": 10}), 10),
])
def test_min_bytes(kernel, elems):
    assert min_bytes(kernel, 4) == elems * 4


#one gemm whose output alone is bigger than the llc
def test_report_counts_kernels_over_llc():
    big = Kernel(0, "gemm", {"m": 2048, "n": 2048, "k": 1})
    small = Kernel(1, "elementwise", {"n": 10}, deps=(0,))
    r = report(Workload("w", "test", 4, (big, small)))
    assert r["kernels"] == 2 and r["over_llc"] == 1
    assert r["total_mb"] * 2**20 == min_bytes(big, 4) + min_bytes(small, 4)
    assert min_bytes(big, 4) > LLC_BYTES