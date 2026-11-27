"""
spike A2.5: one whole workload dag on simgrid with one simple policy
- platform: cpu (4 cores as one host), accel (the 32 x 32 array), mem (its link is the dram link)
- costs are placeholders until the kernel table (B2): flops from the shape, bytes = workloads.size.min_bytes
- policy (fifo): gemm and bgemm on the accel, the rest on the cpu, one kernel per resource at a time,
  lowest ready id first
- checks: every kernel runs once, after all its deps; with unlimited dram the makespan equals a plain
  python replay of the same policy; with real dram it is no shorter, and no shorter than all bytes / bw
run from the repo root: python -m spikes.simgrid_dag data/workloads/cholesky_8x512.json 19.2
(dram GB/s, leave it out for unlimited dram). simgrid runs one simulation per process
"""

import heapq
import sys
from simgrid import ActivitySet, Engine, LinkInRoute, this_actor
from workloads.format import load
from workloads.size import min_bytes

#flop/s, placeholders: gem5's 1-core sgemm x 4 cores; 1024 pes x 2 flop x 1 ghz at scale-sim's 80% util
speeds = {"cpu": 4 * 52.7e9, "accel": 0.8 * 2 * 1024 * 1e9}
fat = 1e15 #links that never limit anything
accel_types = {"gemm", "bgemm"}


def flops(k):
    s = k.shape
    return {"gemm": lambda: 2 * s["m"] * s["n"] * s["k"],
            "bgemm": lambda: 2 * s["b"] * s["m"] * s["n"] * s["k"],
            "potrf": lambda: s["b"] ** 3 / 3,
            "getrf": lambda: 2 * s["b"] ** 3 / 3,
            "trsm": lambda: s["m"] ** 2 * s["n"],
            "syrk": lambda: s["n"] ** 2 * s["k"],
            "elementwise": lambda: s["n"],
            "reduce": lambda: s["n"]}[k.type]()


def resource(k):
    return "accel" if k.type in accel_types else "cpu"


def build(e, dram_bw):
    zone = e.netzone_root.add_netzone_star("soc")
    for name, speed, bw in [("cpu", speeds["cpu"], fat), ("accel", speeds["accel"], fat), ("mem", 1, dram_bw)]:
        host = zone.add_host(name, speed).seal()
        link = zone.add_link("link-" + name, [bw]).seal()
        zone.add_route(host, None, [LinkInRoute(link)], True)
    zone.seal()

#the policy: for each idle resource, the lowest id ready kernel that belongs on it
def pick(waiting, busy, kernels):
    picks = []
    for res in ("cpu", "accel"):
        if res not in busy:
            ready = [kid for kid, deps in waiting.items() if not deps and resource(kernels[kid]) == res]
            if ready:
                picks.append((res, min(ready)))
    return picks


#the same policy with no simgrid and no dram limit: kernel time = flops / speed
#-> kernel id -> (resource, start s, end s)
def run_python(w):
    kernels = {k.id: k for k in w.kernels}
    waiting = {k.id: set(k.deps) for k in w.kernels}
    busy, events, ran, now = {}, [], {}, 0.0
    while waiting or busy:
        for res, kid in pick(waiting, busy, kernels):
            end = now + flops(kernels[kid]) / speeds[res]
            busy[res] = kid
            heapq.heappush(events, (end, res))
            ran[kid] = (res, now, end)
            del waiting[kid]
        now = events[0][0]
        while events and events[0][0] == now:
            _, res = heapq.heappop(events)
            kid = busy.pop(res)
            for deps in waiting.values():
                deps.discard(kid)
    return ran




#the dag on simgrid: one scheduler actor starts kernels as async ptasks and waits for them
#-> kernel id -> (resource, start s, end s)
def run_simgrid(w, dram_bw):
    e = Engine(["simgrid_dag"])
    Engine.set_config("host/model:ptask_L07")
    build(e, dram_bw)
    kernels = {k.id: k for k in w.kernels}
    ran = {}

    def scheduler():
        mem = e.host_by_name("mem")
        waiting = {k.id: set(k.deps) for k in w.kernels}
        busy = {} #resource -> (exec, kernel id, start)
        running = ActivitySet()
        while waiting or busy:
            for res, kid in pick(waiting, busy, kernels):
                k = kernels[kid]
                ex = this_actor.exec_init([e.host_by_name(res), mem], [flops(k), 0],
                                          [0, min_bytes(k, w.dtype_bytes), 0, 0])
                ex.start()
                running.push(ex)
                busy[res] = (ex, kid, Engine.clock)
                del waiting[kid]
            done = [running.wait_any()]
            while (more := running.test_any()) is not None: #everything else that ended at this same time
                done.append(more)
            for ex in done:
                res = next(r for r, (x, _, _) in busy.items() if x is ex)
                _, kid, start = busy.pop(res)
                ran[kid] = (res, start, Engine.clock)
                for deps in waiting.values():
                    deps.discard(kid)

    e.host_by_name("cpu").add_actor("scheduler", scheduler)
    e.run()
    return ran


def check_order(w, ran):
    assert sorted(ran) == sorted(k.id for k in w.kernels), "every kernel runs exactly once"
    for k in w.kernels:
        for d in k.deps:
            assert ran[d][2] <= ran[k.id][1] + 1e-12, f"kernel {k.id} started before its dep {d} ended"


if __name__ == "__main__":
    w = load(sys.argv[1])
    dram_bw = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else fat
    ran = run_simgrid(w, dram_bw)
    check_order(w, ran)
    makespan = max(end for _, _, end in ran.values())
    nbytes = sum(min_bytes(k, w.dtype_bytes) for k in w.kernels)
    on_accel = sum(res == "accel" for res, _, _ in ran.values())
    print(f"{w.name}: {len(ran)} kernels ({on_accel} on accel), makespan {makespan * 1e3:.3f} ms, order ok")
    ideal = max(end for _, _, end in run_python(w).values())
    if dram_bw == fat:
        assert abs(makespan - ideal) <= 1e-9 * ideal, f"simgrid {makespan} vs python replay {ideal}"
        print(f"  unlimited dram: equals the python replay ({ideal * 1e3:.3f} ms)")
    else:
        assert makespan >= ideal * (1 - 1e-9) and makespan >= nbytes / dram_bw * (1 - 1e-9)
        print(f"  {dram_bw / 1e9:g} GB/s: {makespan / ideal:.2f}x the unlimited-dram makespan, "
              f"bytes / bw alone = {nbytes / dram_bw * 1e3:.3f} ms")