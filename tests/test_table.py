from types import SimpleNamespace
from table import table

def test_add_load_lookup(tmp_path):
    path = str(tmp_path / "t.csv")
    row = {"type": "gemm", "shape": {"n": 8, "m": 4, "k": 2}, "resource": "accel", "cores": 1,
           "compute_ns": 10.5, "dram_bytes": 256, "source": "test"}
    table.add([row], path)
    table.add([dict(row, compute_ns=20.0)], path) #same key: replaced, not duplicated
    t = table.load(path)
    assert list(t) == [("gemm", "k=2;m=4;n=8", "accel", 1)]
    kernel = SimpleNamespace(type="gemm", shape={"m": 4, "n": 8, "k": 2})
    assert table.lookup(t, kernel, "accel") == (20.0, 256)

def test_distinct_covers_the_suite():
    kinds = table.distinct()
    assert len(kinds) == 48 and len({(t, table.shape_key(s)) for t, s in kinds}) == 48