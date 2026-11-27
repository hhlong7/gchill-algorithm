how these scripts connect to the tools to get results:

spikes/ => tools/ => results (one chain per tool)

for simgrid, we dont write anything, basically we are chekcing to see if simgrid behaves before we implement its backend and the system view

simgrid_ptask.py ─> import simgrid ─> runs kernels on a made-up platform ─> prints ok / times
simgrid_rate.py ─> imports build() from simgrid_ptask ─> changes link bandwidth mid run ─> prints

simgrid_ptask.py is for how a kernel is modeled, as it builds a small soc in simgrid with host cpu0, cppu1 and mem. mem's link is the dram at 10gb/s
a kernel is one parallel exec
=> kernel time = max(compute, bytes / bandwidth) and the kernels running together split the dram link evenly => mapping what we want to implement for the simgrid backend

simgrid_rate.py is to model if the arbiter can steer a running kernel

so what that means is that it would start the 10 ms memory bound kernel and changes it partway thruough.
=> a kernel's dram bandwidth can change while it runs, => so the engine recal its finish time. => maps the loss curve need from the dram simulator. and also the ability to pause and resume work, in case the algorithm needs them



scalesim:

scalesim_gemm.py: the accelerator to results/scalesim/

it would write a csv file: gemm, 256, 256, 256. and then runs scalesim with the configs in configs/ as a 32x32 array output, 1ghz

=>
COMPUTE_REPORT.csv

Total Cycles	Stall	Compute Util	incl. prefetch
20,351	        0	    80.5%	        24,138

The best possible is 256³ ÷ 1024 PEs = 16,384 cycles, so this is 1.24x that.
Where 80.5% comes from: the output is 8 x 8 = 64 tiles of 32x32. Each tile takes K = 256 cycles plus about 62 cycles to fill and drain the array. 64 x 318 = 20,352, and 256 ÷ 318 = 80.5%.

Stall = 0 because bandwidth is set to calculated, which means scalesim assumes DRAM never makes the array wait. thats on purpose: SimGrid adds the DRAM time later.


DETAILED_ACCESS_REPORT.csv (DRAM, in words)

	                Words	vs. size of the matrix
IFMAP (A) reads	    524,288	    8x A
Filter (B) reads	81,884	    1.25x B
OFMAP (C) writes	65,567	    1x C
Total	            671,739	    3.4x the minimum 196,608

A is read 8 times, once for each 32-column block of B. The IFMAP SRAM is 64k words, and double buffering leaves only half for A, so A doesn't fit.

=> accel: 20us of compute, but 2.69 mb / 19.2 gb/s => 140 us of dram time
where the cpu is about 173us of 4 cores, needing only about 2.4 gb/s

=> this is a rough comparison, so this means that the accel is limited by the dram, not compute, this ultimately means that the choice of computation device, and dram contention decide the outcome


for raumlator: dram under sharing to results/

it is for doing for n = 1,2,4,8,16,32 streams and 3 seeds, write streams_<n>_<seed>.trace 
-each stream starts on a random 4kb page and reads 64b lines in order

=>
More streams cause more row conflicts in the DRAM banks, so the total bandwidth drops. It isn't just split more ways
At 32 streams, all the kernels together get only 7.7 gb/s
This is the loss curve from the dram simulator will produce, gchill relies on it: running fewer memory-heavy kernels at the same time can finish sooner


for gem5: the cpu to results/

gem5_gemm.c is the program gem5 runs, where it fills a, b and c, then it runs a 8**3 warm up so the openblas starts its threads. after that scrub() reads a 32mb buffer twice, leaves nothing in the caches, then reset the counters, run the cblas_sgemm, then dump reset, scrub again, this time forces c dirty lines out to dram, and then dump.

gem5_se.py is the machien description: neoverse v2 cores at 3ghz, own l1 and l2
-running on ideal mem, which means 0 latency and unlimited bandwidth, and ddr which is a shared 12 mib llc + ddr4-2400

gem5_gemm.py is the drive where it would compiles the c file in gem5 container, then output result to results/. runs gem5 4 times and each run 


simgrid_dag.py is splited into 3 steps: costs and platform, the policy and a python replay of it, and then simgrid running and verificaiton

for the first part (cost and platform): it is the file header, with the placeholder costs,
containing flops, bytes, and speeds, from each kernel's shape, workloads.size.min_bytes, and from cpu and accelerator

for running the check:

on mac

.venv/bin/python -c "from spikes.simgrid_dag import flops, resource, speeds; from workloads.format import load; w = load('data/workloads/cholesky_8x512.json'); [print(k.id, k.type, resource(k), f'{flops(k) / speeds[resource(k)] * 1e6:.1f} us') for k in w.kernels[:4]]"

basically, this step will output the number that will tell us which workloads become dram limited when they share the bus, for ex the dnns go 2-4.5x slower at ~20gb/s while chol8x512 doesnt, since the cpu is the limiting factor

then we implement a simple fifo to the engine to see if we can have a python scheduler driving simgrid through a real dag workload (it dispatches ready kernels, waits for one to finish, collects any others that finished at the same moment, unlocks their children, and repeats.)

to check with the fifo: 

on mac

.venv/bin/python -c "from spikes.simgrid_dag import run_python; from workloads.format import load; ran = run_python(load('data/workloads/cholesky_8x512.json')); print(len(ran), 'kernels, makespan', round(max(e for _, _, e in ran.values()) * 1e3, 3), 'ms'); [print(kid, ran[kid][0], round(ran[kid][1] * 1e6, 1), round(ran[kid][2] * 1e6, 1)) for kid in (0, 1)]"

=> output: 
120 kernels, makespan 37.353 ms
0 cpu 0.0 212.2
1 cpu 212.2 848.9

this means that kernel 0 (potrf) runs from 0 to 212 us then kernel 1 trsm depends on it so it picks up from there


then we are going to run the dag in simgrid with the checks, with the func run_simgrid()

it would build the platform and starts one scheduler, which loops until the dag is done

to check:

.venv/bin/python -m spikes.simgrid_dag data/workloads/cholesky_8x512.json
.venv/bin/python -m spikes.simgrid_dag data/workloads/cholesky_8x512.json 19.2
.venv/bin/python -m spikes.simgrid_dag data/workloads/bert_base_s128_b1.json
.venv/bin/python -m spikes.simgrid_dag data/workloads/bert_base_s128_b1.json 19.2


outputs:

cholesky_8x512: 120 kernels (56 on accel), makespan 37.353 ms, order ok
  unlimited dram: equals the python replay (37.353 ms)
cholesky_8x512: 120 kernels (56 on accel), makespan 37.353 ms, order ok
  19.2 GB/s: 1.00x the unlimited-dram makespan, bytes / bw alone = 19.224 ms
bert_base_s128_b1: 176 kernels (97 on accel), makespan 13.699 ms, order ok
  unlimited dram: equals the python replay (13.699 ms)
bert_base_s128_b1: 176 kernels (97 on accel), makespan 29.369 ms, order ok
  19.2 GB/s: 2.14x the unlimited-dram makespan, bytes / bw alone = 29.369 ms