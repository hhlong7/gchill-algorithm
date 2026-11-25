"""
spike A2.4: can the arbiter change a running kernel's dram rate without restarting it?
1. throttle: a memory bound kernel (1e8 bytes at 10 GB/s = 10 ms) runs 5 ms, then its core's link drops
   to 5 GB/s => the other 5e7 bytes take 10 ms => 15 ms in all
2. pause: same kernel, suspended at 5 ms, resumed at 8 ms => 13 ms in all
3. half done: at 5 ms remaining_ratio is 0.5 (the backend reads this for the fraction left)
"""

import sys
from simgrid import Engine, Link, this_actor
from simgrid_ptask import build, fat

results = {}


def start_kernel():
    e = Engine.instance
    hosts = [e.host_by_name("cpu0"), e.host_by_name("mem")]
    task = this_actor.exec_init(hosts, [1e5, 0], [0, 1e8, 0, 0])
    task.start()
    return task


def tests():
    link = Link.by_name("link-cpu0")

    start = Engine.clock
    task = start_kernel()
    this_actor.sleep_for(5e-3)
    results["half_done"] = (task.remaining_ratio, 0.5)
    link.set_bandwidth(5e9)
    task.wait()
    results["throttle"] = (Engine.clock - start, 15e-3)
    link.set_bandwidth(fat)

    start = Engine.clock
    task = start_kernel()
    this_actor.sleep_for(5e-3)
    task.suspend()
    this_actor.sleep_for(3e-3)
    task.resume()
    task.wait()
    results["pause"] = (Engine.clock - start, 13e-3)


if __name__ == "__main__":
    e = Engine(sys.argv)
    Engine.set_config("host/model:ptask_L07")
    build(e)
    e.host_by_name("cpu0").add_actor("tests", tests)
    e.run()
    for name, (got, want) in results.items():
        ok = abs(got - want) <= 1e-9 * max(want, 1)
        print(f"{name:10s} got {got:.6e}  want {want:.6e}  {'ok' if ok else 'MISMATCH'}")