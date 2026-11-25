from collections import Counter
import pytest
from workloads.chameleon import load_rec
from workloads.format import validate

#potrf -> a starpu control task -> trsm, a dep on a job outside the trace, job ids out of submission order
REC = """Name: spotrf
Model: spotrf
JobId: 7
SubmitOrder: 1
DependsOn: 3
Parameters: M4x4x6

Name: _starpu_sync_task
JobId: 9
DependsOn: 7

Name: strsm
Model: strsm
JobId: 5
SubmitOrder: 2
DependsOn: 9
Parameters: M4x4x6 M4x4x6
"""


def rec_file(tmp_path, text):
    path = tmp_path / "t.rec"
    path.write_text(text)
    return str(path)


def test_control_tasks_and_ghost_deps(tmp_path):
    w = load_rec(rec_file(tmp_path, REC), "t")
    assert [(k.type, k.shape, k.deps) for k in w.kernels] == [
        ("potrf", {"b": 4}, ()), #its dep on job 3 is outside the trace: dropped
        ("trsm", {"m": 4, "n": 4}, (0,)), #waited on the sync task, which waited on potrf
    ]
    assert (w.source, w.dtype_bytes) == ("chameleon", 4)


def test_rejects_edge_tiles(tmp_path):
    with pytest.raises(NotImplementedError, match="square"):
        load_rec(rec_file(tmp_path, REC.replace("M4x4x6 M4x4x6", "M4x4x6 M4x3x6")), "t")


def test_rejects_unknown_codelet(tmp_path):
    with pytest.raises(NotImplementedError, match="sgeqrt"):
        load_rec(rec_file(tmp_path, REC.replace("Name: strsm", "Name: sgeqrt")), "t")


#most kernels on any path from the first task to the last
def longest_chain(w):
    depth = {}
    for k in w.kernels: #kernel ids follow submission, so deps come first
        depth[k.id] = 1 + max((depth[d] for d in k.deps), default=0)
    return max(depth.values())


#the real traces: textbook task counts, one entry task, and the 3t - 2 critical path for t x t tiles
@pytest.mark.parametrize("name, tiles, counts", [
    ("cholesky_8x512", 8, {"potrf": 8, "trsm": 28, "syrk": 28, "gemm": 56}),
    ("cholesky_16x256", 16, {"potrf": 16, "trsm": 120, "syrk": 120, "gemm": 560}),
    ("lu_8x512", 8, {"getrf": 8, "trsm": 56, "gemm": 140}),
])
def test_traces(name, tiles, counts):
    w = load_rec(f"data/chameleon/{name}.rec", name)
    validate(w)
    assert Counter(k.type for k in w.kernels) == counts
    assert [k.id for k in w.kernels if not k.deps] == [0]
    assert longest_chain(w) == 3 * tiles - 2