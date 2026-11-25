# Open-source stack and plan for gchill

Handoff document for a new Claude Code session. It replaces `OPEN_SOURCE_INTEGRATION.md`: that file is the
older plan (SimGrid for everything, gem5 for whole DAGs, no SCALE-Sim) and is out of date.

Last updated 2026-11-24, at commit `bc6000d` on branch `heurisitics`.

---

## 0. Read this first: working rules

These come from the user and override any default behavior.

- **The user writes the project code.** Do not create or edit project files, install packages or run setup
  commands unless the user explicitly says so ("fix it for me", "write it"). Break each phase into small
  steps, explain each one, and **show the code in chat** for the user to type. Reviewing or running their
  code to check it is fine.
- **The user runs git.** Never commit, merge or push. Show the commands, one command per ```` ```bash ````
  block.
- **No attribution lines.** No `Co-Authored-By` and no "Generated with Claude" in commits or PRs.
- No emojis. No ADRs. Keep `.vscode/` out of commits.
- `brain/`, `.agents/`, `.claude/` and `CLAUDE.md` are local only (listed in `.git/info/exclude`). Never
  `git add` them.
- **The partner owns** `sim/policies/eshcs.py` and `sim/policies/optimal.py`. Changes that touch them, such
  as switching them to the system view, should be agreed with the partner.
- Per `CLAUDE.md`, read the relevant `brain/` notes before acting. The most relevant one is
  `brain/findings/gchill-contention-dram-priority.md`. The brain plan `brain/plans/02-open-sim-and-workloads/`
  is out of date: this document supersedes it.

---

## 1. Project in one paragraph

CPE 5300 (graduate computer architecture) research project. A CPU+accelerator SoC (N CPU cores plus a
systolic-array accelerator) shares one DRAM. A DAG of kernels (GEMMs and others) must be scheduled onto it,
and kernels running at the same time slow each other down by competing for DRAM bandwidth.

**The contribution is one algorithm, gchill** (`sim/policies/gchill/`): a contention-aware task scheduling algorithm for the SoC system that also tells the memory controller which running kernel is most critical.

**Everything else is a baseline:**
- HEFT (`heft.py`);
- ESHCS (Xu 2023, `eshcs.py`, partner's);
- the A*/DFBnB optimal search (`optimal.py`, partner's), used as a reference;
- eft_greedy (StarPU dmda-style, `heft_greedy.py`), Qilin-style, contention-aware, FIFO.

**Goal of this plan:** every part of the test system is an open-source tool. The simulator engine, the CPU,
the accelerator, the DRAM and the workloads all come from outside tools. The team writes only gchill, the
glue code and the measurement.

---

## 2. Final stack

| Part | Tool | How it's used |
|---|---|---|
| **Simulator (engine)** | **SimGrid** (Python S4U bindings) | Runs every experiment: N core hosts, one accelerator host, one memory host, and a shared DRAM link. Each policy's `pick()` is called from a SimGrid actor |
| **CPU** | **gem5** (ARM out-of-order core, SE mode) running **OpenBLAS** kernels | Offline, once per kernel shape: compute time at 1..N cores and DRAM bytes, cached in the kernel table |
| **Accelerator** | **SCALE-Sim** (`pip install scalesim`) | Offline, once per GEMM shape: compute cycles and DRAM bytes, cached in the kernel table |
| **DRAM** | **Ramulator 2** | Offline: delivered bandwidth vs how many kernels share DRAM (the loss curve), used in SimGrid's sharing callback. Optional full-DAG replays as a check |
| **Workloads** | **torch.fx** graphs of open models (torchvision / HuggingFace); **Chameleon / StarPU** tiled Cholesky, LU and QR; **STG** (Kasahara) | Imported into JSON file and load into the simulator. STG has no kernel types or byte counts, so it is used only to check the optimal search, not as a contention workload |
| **Reference baselines** | **SAGA** (`pip install anrg.saga`) | Reference HEFT/CPoP code to check our HEFT against |
| **Algorithm (ours)** | **gchill** | Stays in Python behind the existing `Policy` interface |


**Our current simulator** (`sim/simulator.py`) stays as a fast development testbed and as the reference for
checking the SimGrid backend. It is not part of gchill.

---

## 3. Architecture

```
offline, once per kernel shape
  gem5 (CPU)        -> compute ns at 1..N cores, DRAM bytes   \
  SCALE-Sim (accel) -> compute ns, DRAM bytes                  > kernel table
  Ramulator 2       -> DRAM loss vs how many kernels share it  -> loss curve

every run
  workload file -> SimGrid backend --view--> policy.pick(view) -> picks
                       |  ^                       (gchill, heft, eshcs, ...)
                       v  |
                    SimGrid engine: core hosts, accel host, DRAM link
                    (sharing callback = loss curve + DRAM arbiter)
```

### 3.1 Workload file (new)

One format for every source, so importers and runs are independent. Suggested JSON:

```
{ "name": "cholesky_8x128", "source": "chameleon", "dtype_bytes": 2,
  "kernels": [ { "id": 0, "type": "potrf", "shape": {"b": 128}, "deps": [], "arrival_ns": 0.0 },
               { "id": 1, "type": "gemm",  "shape": {"m": 128, "k": 128, "n": 128}, "deps": [0], ... } ] }
```

- `type` is one of a small fixed set: `gemm`, `potrf`, `trsm`, `syrk`, `elementwise`, `reduce`, ...
- `shape` is whatever that type needs to look up its cost.
- A loader turns a file into the existing `Kernel` objects (`sim/kernels.py`), so the current simulator and
  the tests keep working.
- Size the suite well above the 12 MB LLC. Today's Xu DAGs move only 0.4-4.8 MB, which is why contention
  barely shows.

### 3.2 Kernel table (new)

A lookup with key `(kernel type, shape, resource, cores)` and value `(compute_ns, dram_bytes, source)`.
- `resource` is `cpu` or `accel`; `cores` is 1..N for the CPU and 1 for the accelerator.
- `source` is `gem5` or `scalesim`, plus a hash of the tool's config.
- Store it as CSV or parquet under `data/`.

| Column filled by | How |
|---|---|
| **gem5: CPU compute_ns** | Run the OpenBLAS (or reference) kernel on an ARM out-of-order core in SE mode with ideal memory, so only compute is timed. Run it at k = 1..N threads for multicore scaling |
| **gem5: CPU dram_bytes** | The same kernel with a normal cache hierarchy (L1/L2/LLC + DDR): LLC misses x line size |
| **SCALE-Sim: accel compute_ns** | Compute cycles (without memory stalls) x the clock period |
| **SCALE-Sim: accel dram_bytes** | DRAM read + write bytes from its reports |

- Both the SimGrid backend and gchill's cost model read this table. Standalone costs then agree everywhere,
  and only the contention model differs between the testbed and the policies.
- `sim/policies/cost_model.py` (`segments`) and the policies' cost estimates switch from the `model.py`
  formulas to table lookups. Keep the old formulas only as a fallback for tests.

### 3.3 Loss curve (new)

`delivered_fraction(n)`: the fraction of peak DRAM bandwidth actually delivered when n kernels stream at
the same time.
- **Measure it** with Ramulator 2: n concurrent streams (sequential read/write mixes in separate address
  regions), repeated for each n.
- **Store** the points plus the DRAM config (standard, channels, ranks).
- It plugs into the existing `degradation(n_agents)` hook (`sim/memory.py`) and into SimGrid's
  nonlinear sharing callback.
- Optional check: replay a few whole DAGs' memory traffic through Ramulator 2 and compare with the curve.

### 3.4 System view (new)

What a policy may read, and the only thing it ever sees. It holds the fields the policies read from `sim`
today:

| Field | Today in `sim` |
|---|---|
| `now` | `sim.now` |
| `kernels`, `by_id`, children | `sim.kernels`, `sim.by_id` |
| running jobs: kernel, resource, cores, start, fraction left, current bandwidth | `sim.jobs` |
| free cores, which kernel is on each core | `sim.cpu.free_cores()`, `sim.cpu.cores` |
| accelerator free, its current kernel | `sim.accelerator.free`, `.current` |
| config (cores, peak DRAM bandwidth) | `sim.config` |
| kernel table | new |

- Both the current simulator and the SimGrid backend produce a view, so every policy runs on both
  unchanged.
- `sim.degradation` is **not** in the view. Policies must not see the testbed's true DRAM model. If a
  policy needs a DRAM model (only the `delay` / `harm` / `pairing` / `feed` / `adjusted` flags use
  `what_if`), it carries its own simple one: proportional share, no loss.

### 3.5 SimGrid backend (new, about 200-300 lines)

- **Platform:** N core hosts, an `accel` host and a `mem` host, all joined to `mem` by one DRAM link with
  bandwidth = peak.
  - Host speed is 1e9 "flops"/s, so a kernel's work equals its `compute_ns` from the table.
- **Each kernel becomes one parallel task:** work on its host(s) plus `dram_bytes` between its host and
  `mem`. SimGrid's ptask model (L07) then advances it at the pace of its slowest resource, i.e.
  max(compute, bytes / bandwidth), with SimGrid doing the sharing.
- **A malleable CPU kernel** is one parallel task over k core hosts, using the table's k-core time.
- **Scheduler actor loop** (sketch; exact SimGrid Python names to confirm in the spike):

```python
def scheduler(policy, workload, view):
    policy.on_submit(workload.kernels, view)
    while not workload.all_done():
        view.refresh()                          # now, running, free cores, fractions left
        for k in workload.newly_ready(view.now):
            policy.on_queued(k, view)
        t0 = time.perf_counter_ns()
        picks = policy.pick(view)               # [(kernel, cpu|accelerator, cores)]
        view.sched_ns += time.perf_counter_ns() - t0
        for kernel, kind, cores in picks:
            launch(kernel, kind, cores)         # starts a ptask, records it in the view
        apply_dram_arbiter(policy, view)        # bandwidth per running kernel, section 3.6
        wait_first(running_tasks, next_arrival(workload))   # a finish or the next release
```

- **Decision cost** (µs per pick) is measured on the host and reported next to makespan, not charged to
  simulated time. Optionally charge it by having the actor wait the measured time before launching.
- **Outputs:** the same per-run row as `runs/gchill_sweep.py` `run_one` (makespan, lower bound, gap, DRAM
  and accelerator utilization, µs per pick, ...), so `compare()`, `gap_table()` and `e4_table()` work
  unchanged.

### 3.6 DRAM loss and arbiter

At every event the backend decides how much DRAM bandwidth each running kernel gets:

1. **Order:** if the policy defines `bw_order(view)`, use that order. Only gchill does. Otherwise use
   **oldest-first**. HEFT may optionally use its own upward rank (measured before: it hurts HEFT under
   lossless DRAM).
2. **Allocate in that order, with loss counted only over served kernels, and stop admitting kernels when
   one more would cost the served ones more than it gets.** This is the prototype, which lives only in
   session scratch and is reproduced here for typing into `sim/memory.py`:

```python
def allocate_priority(demands, memory, order, degradation=no_degradation):
    peak = memory.dram_bw_bytes_per_s
    allocation = {agent: 0.0 for agent in demands}
    served = []
    for agent in order:
        want = min(demands[agent], peak)
        if want <= 0:
            continue
        available = peak * degradation(len(served) + 1)
        used = sum(allocation[a] for a in served)
        if used > available: #the new agent would cost the ones already served more than it gets
            break
        allocation[agent] = min(want, available - used)
        served.append(agent)
        if allocation[agent] < want:
            break
    return allocation
```

3. **The current simulator** gets the same hook in `reallocate()`:
   - use `allocate_priority` with `policy.bw_order(...)` if the policy has it, else the default order;
   - a kernel that gets 0 bandwidth is starved until the next event (no finish event scheduled), instead
     of raising an error.
4. **In SimGrid**, apply the allocation as a per-task bandwidth limit on the DRAM link. **Open question
   for the spike:** can a running task's limit be changed, or must the backend cancel it and relaunch the
   remaining work with the new limit? Either works, because the backend knows each task's remaining
   fraction.
5. **Optional, more realistic:** the same priority rule as a Ramulator 2 scheduler plugin (C++) for the
   full-DAG replays.

### 3.7 gchill on this stack (final form)

| Piece | Status |
|---|---|
| Upward rank (`priority.py`), costs from the kernel table | keep |
| Fastest-alone plan + online greedy dispatch in rank order | keep |
| Override: use the other resource if it finishes sooner than waiting | keep |
| Malleable cores: spare cores go to the CPU picks that gain most | keep |
| **Live DRAM priority (gchill v2):** `bw_order(view)` sorts running kernels by live criticality = time left at peak bandwidth on its actual resource + the largest upward rank among its children | **new, the key mechanism** |
| Delay + trigger | **open decision** (section 7). Proposed: drop them from the preset and keep the flags only for the E4 ablation |
| ILP plan, adjusted costs, harm, pairing, feed, replan | flags kept for the E4 ablation only, off in the preset |

- **Why live criticality:** an earlier version ranked running kernels by their static upward rank. After an
  override moves a kernel to the other resource, that rank is stale, and the critical kernel was starved
  (worst case +40%). Recomputing "time left on the actual resource + max child rank" fixed it.
- **Live criticality prototype** (scratch, against the current simulator's `job` fields):

```python
def crit(self, kid):          # running kernel id -> ns of work still hanging on it
    job = sim.jobs[kid]
    peak = sim.config.memory.dram_bw_bytes_per_s
    left = job.left * job.segment.duration_ns(peak) + sum(s.duration_ns(peak) for s in job.segments[job.idx + 1:])
    return left + max((self.up[c] for c in children[kid]), default=0.0)   # self.up = upward_rank in ns

def bw_order(self, kids):
    return sorted(kids, key=lambda k: -self.crit(k))
```

  On the view, `left` becomes `fraction_left x the table's standalone time`.

- **Proposed preset:** `best = dict(malleable=True)` plus the DRAM priority, which is on in the final gchill.

---

## 4. Baselines and fairness rules

- Every policy gets the same view, the same kernel table, the same workloads and the same testbed.
- Every policy gets a reasonable DRAM arbiter: oldest-first by default. Only gchill supplies criticality,
  because that's gchill's idea. Report HEFT + oldest-first and HEFT + its own rank as separate rows.
- **Textbook HEFT** (Topcuoglu 2002): done in `sim/policies/heft.py`, with tests in `tests/test_heft.py`.
  Check it against SAGA's HEFT on a few DAGs with contention off.
- **Moldable HEFT:** to do. HEFT that may give a CPU kernel k cores, using the table's k-core times.
- **ESHCS** and the **optimal search** run through the view unchanged, apart from renaming `sim` fields.
  - The optimal search plans with our model (kernel table + simple DRAM), so on the testbed it is a
    **reference schedule, not a proven bound**. gchill beating it on SimGrid is a finding, not a bug.
  - The true floor is a lower bound computed from the kernel table.
  - Optional: check the search against STG's published optimal schedule lengths (accelerator and
    contention off).

---

## 5. Evidence so far

All of this is on the current simulator with session-scratch code (not in the repo), on Xu-style DAG
families plus a scratch tiled Cholesky. DRAM loss model: delivered = peak x (1 - s(n - 1)).

| Comparison | Result |
|---|---|
| Textbook HEFT vs ESHCS (Xu families, lossless) | HEFT 3.35% faster |
| gchill 3-piece (override + upward + malleable) vs textbook HEFT | -0.43% mean, slower on 38% of DAGs: about a tie |
| gchill + live DRAM priority vs textbook HEFT | **-3.6%** (lossless), **-16.5%** (s = 0.1) |
| gchill + live DRAM priority vs HEFT given the same arbiter | **-3.6% / -6.1%**, faster on 83-94% of DAGs |
| HEFT + arbiter vs plain HEFT, lossless | +1.3% to +4.6% worse: a fixed plan can't use the arbiter |
| gchill + live DRAM priority vs 3-piece gchill | -3.2% (lossless), up to -14.5% (s = 0.1) |
| Arbiter order controls | Oldest-first gets most of the gain under loss; criticality adds about 1-1.5% and halves the worst case |
| Tiled Cholesky, gchill vs HEFT | about -23%: about 9 points from ranking by planned resource, the rest from malleable POTRF. **Depends on the placeholder CPU speed** |
| Priority rankings tested (upward, lower_bound, eshcs, heft_mean, cpop, oct, desc) | Keep upward; differences under about 1% |

**Caveats the new stack must address:**
- **CPU speed is a placeholder:** 10 GFLOPS per core vs a 2.05 TFLOPS array, about 200x. A real core with
  SIMD makes it about 20-40x.
- The workloads fit in the LLC.
- The policies' cost model and the testbed were the same model.

---

## 6. Phases

Hours are rough typing-plus-debugging estimates for the user. Phases 2-4 need no SimGrid and can start
while the Phase 1 spikes run.

| # | Phase | Deliverable | Verify | Hours |
|---|---|---|---|---|
| 0 | **Remove rollout** | Delete `sim/policies/gchill/rollout.py`; remove `rollout_k`, `n_rollouts`, `best_rollout`, `greedy(first=, widen=)` in `ilp.py`; the `g_roll*`, `gchill_rollout`, `e4_roll*` rows and `skip` in `runs/gchill_sweep.py`; the rollout tests in `tests/test_gchill.py` and `tests/test_gchill_sweep.py` | `grep -rn -E "rollout\|play_out\|n_rollouts" sim runs tests` finds nothing; pytest green | 1-2 |
| 1 | **Spikes**, 1-2 days each | (a) SimGrid on macOS (source build or Homebrew) and on the partner's Windows machine (WSL): one DAG, one policy. (b) gem5 (Docker on macOS): one OpenBLAS GEMM in SE mode, plus a multithreaded run. (c) SCALE-Sim: one GEMM. (d) Ramulator 2: one trace | Each tool produces a number for the same GEMM; note install steps in `brain/` | 8-16 |
| 2 | **Workload format + importers** | Loader to `Kernel`; importers for torch.fx (2-3 open models), Chameleon/StarPU tiled Cholesky/LU/QR, STG; a suite with data well above the LLC | Round-trip tests; DAGs are acyclic; byte totals vs LLC printed | 20-35 |
| 3 | **Kernel table** | gem5 runner (CPU, 1..N cores), SCALE-Sim runner (accelerator), table file, table-based `cost_model.py` | Table covers every kernel in the suite; spot-check against the roofline; current simulator runs on table costs | 20-35 |
| 4 | **System view** | View class; current simulator produces it; gchill, HEFT, eft_greedy, Qilin, contention-aware and FIFO read it (ESHCS and optimal together with the partner) | Every policy gives the same makespans as before on the old model | 8-15 |
| 5 | **SimGrid backend, no contention** | Platform, ptasks, scheduler actor, run row | With unlimited DRAM, every policy's makespan equals the current simulator's on the same table | 25-45 |
| 6 | **Ramulator loss curve** | Stream traces, curve fit, plugged into `degradation` and SimGrid's sharing callback | Curve is monotone, in (0, 1]; small cases match a hand calculation | 15-25 |
| 7 | **DRAM arbiter + gchill v2** | `allocate_priority` in `sim/memory.py` and the simulator hook; arbiter in the backend; `bw_order` with live criticality in gchill; oldest-first default | Unit tests on `allocate_priority`; reproduce the section 5 numbers on the current simulator first, then on SimGrid | 10-20 |
| 8 | **Baselines** | Moldable HEFT; SAGA HEFT check; ESHCS and optimal on the view | HEFT matches SAGA with contention off | 10-20 |
| 9 | **Experiments** | Sweeps over suite x bandwidth x loss curve; `compare()` gchill vs each baseline; E4 ablation of gchill's pieces; decision cost table | Paired win/loss and worst case reported per bandwidth and per family | 15-25 |
| opt | Ramulator priority-scheduler plugin + full-DAG replays; STG check of the optimal search | | | 35-60 |

**Core total:** about 130-240 h.

**Stopping points:**
- **After 7 on the current simulator:** real kernel costs, real loss curve, gchill v2. A credible result
  without SimGrid.
- **After 9:** the full open-source testbed.

---

## 7. Open decisions

1. **Delay + trigger in gchill's preset?** Proposed: drop them, keeping `best = dict(malleable=True)` +
   DRAM priority. Then gchill needs no DRAM model of its own, and its contention handling comes entirely
   from the memory-controller hint.
2. **Precision:** the accelerator is bf16 (`dtype_bytes = 2`). OpenBLAS has bf16 GEMM only on some ARM
   targets. Pick one precision for both sides, or document the mismatch.
3. **Non-GEMM kernels on the accelerator:** SCALE-Sim models GEMM. Do TRSM/SYRK map to GEMM shapes
   (approximation), or do they run only on the CPU?
4. **CPU model in gem5:** which ARM out-of-order configuration (core width, caches, clock). Use one
   published configuration and cite it.
5. **HEFT's arbiter row:** oldest-first only, or also its own rank.
6. **Deadline:** decides whether to stop after Phase 7 or Phase 9.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| SimGrid has no macOS wheel; Windows needs WSL | Phase 1 spike first; fall back to the current simulator + real tables + Ramulator curve (stopping point 1) |
| Changing bandwidth limits on running SimGrid tasks | Cancel and relaunch with the remaining work |
| gem5 multithreaded SE runs are fragile | Use gem5's ready-made full-system Linux image for the multicore runs only |
| A real CPU speed shrinks the Cholesky gain | That's the point of measuring it. Report what remains |
| Small statistical base | Many DAGs per family; paired comparisons; report win rate and worst case, not only the mean |

---

## 9. Links

- SimGrid: https://simgrid.org/doc/latest/ (Python examples: https://github.com/simgrid/simgrid/tree/master/examples/python)
- gem5: https://www.gem5.org/documentation/
- OpenBLAS: https://github.com/OpenMathLib/OpenBLAS
- SCALE-Sim: https://github.com/scalesim-project/SCALE-Sim
- Ramulator 2: https://github.com/CMU-SAFARI/ramulator2
- Chameleon: https://gitlab.inria.fr/solverstack/chameleon, StarPU: https://starpu.gitlabpages.inria.fr/
- STG: https://www.kasahara.cs.waseda.ac.jp/schedule/
- SAGA: https://github.com/ANRGUSC/saga
- Gables (HPCA 2019, roofline for SoCs with shared DRAM): https://research.cs.wisc.edu/multifacet/papers/hpca19_gables.pdf
- HEFT: Topcuoglu, Hariri, Wu, IEEE TPDS 2002
