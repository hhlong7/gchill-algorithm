these are the DAGs referenced from format.py

=> so that every workload is in the same shape, no matter which open source tool it came from
eg: 
{name: cholesky_8x512, source: chameleon, dtype_bytes: 4, kernels: [
  {id: 0, type: potrf, shape: {b: 512},        deps: [],  arrival_ns: 0},
  {id: 1, type: trsm,  shape: {m: 512, n: 512}, deps: [0], arrival_ns: 0}, ...]}


there are 5 single workloads files
-resnet
-bert => both of them are from torchfx
-chol8x512, chol16x256
-lu8x512 => three of these are from chameleon

3 mixes workload files: single workloads join into one DAG with renumbered ids

=> the problem is that a kernel only says what to compute, its type, sizes, and what it waits for (dependencies), but it doesn't know what time or what device(cpu/gpu) that it will be run on

=> so that in phase b (integration) we are adding kernel table to record time, and the gchill algorithm :)))))
