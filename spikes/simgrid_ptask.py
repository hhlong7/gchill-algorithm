"""
spike A2: does simgrid's ptask model (L07) do what the backend needs?
1. one kernel = compute on its core + bytes to mem => it takes max(flops / speed, bytes / dram bw)
2. two memory bound kernels at once share the dram link => each takes twice as long
star zone: every host has its own link to the center, mem's link is the dram link, the core links are
so fast they never limit anything
"""

import sys
from simgrid import Engine, LinkInRoute, this_actor

speed = 1e9 #flops/s, so 1 flop of work = 1 ns
dram_bw = 10e9 #bytes/s
fat = 1e15 #core links, never the bottleneck


def build(e):
    zone = e.netzone_root.add_netzone_star("soc")
    for name, bw in [("cpu0", fat), ("cpu1", fat), ("mem", dram_bw)]:
        host = zone.add_host(name, speed).seal()
        link = zone.add_link("link-" + name, [bw]).seal()
        zone.add_route(host, None, [LinkInRoute(link)], True)
    zone.seal()


#one kernel as a ptask: flops on its core, bytes from its core to mem
def kernel(core, flops, nbytes):
    e = Engine.instance
    hosts = [e.host_by_name(core), e.host_by_name("mem")]
    start = Engine.clock
    this_actor.parallel_execute(hosts, [flops, 0], [0, nbytes, 0, 0])
    return Engine.clock - start


results = {}


#1. alone, one after another: compute bound, memory bound, balanced
def alone():
    for name, flops, nbytes in [("compute", 2e6, 1e6), ("memory", 1e5, 1e8), ("balanced", 1e7, 1e8)]:
        took = kernel("cpu0", flops, nbytes)
        want = max(flops / speed, nbytes / dram_bw)
        results[name] = (took, want)


#2. after the alone runs: two memory bound kernels at the same time, one per core
def together(core):
    this_actor.sleep_for(1.0) #wait till the alone runs are done
    results["shared_" + core] = (kernel(core, 1e5, 1e8), 2 * 1e8 / dram_bw)


if __name__ == "__main__":
    e = Engine(sys.argv)
    Engine.set_config("host/model:ptask_L07") #required for ptasks, before the platform
    build(e)
    e.host_by_name("cpu0").add_actor("alone", alone)
    e.host_by_name("cpu0").add_actor("together0", lambda: together("cpu0"))
    e.host_by_name("cpu1").add_actor("together1", lambda: together("cpu1"))
    e.run()
    for name, (took, want) in results.items():
        ok = abs(took - want) <= 1e-9 * max(want, 1)
        print(f"{name:10s} took {took:.6e} s  want {want:.6e} s  {'ok' if ok else 'MISMATCH'}")