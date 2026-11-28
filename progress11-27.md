implemented the kernel table

the kernel table is the one place where it take care of how long does this kernel take, and how much dram traffic does it make. the open source tools fill it in once, and then everything that runs later which is simgrid and the policies reads from it


the policy would be built on this data:
-rank system: upward rannk, would be based from the time row in the table (the longest remaining path)
-cpu vs accel: comparing cpu row against accel row
-multicore or not: 1-4 core row to see if more cores help
-setting dram priority: dram bytes / compute ns => how much bw a kernel wants + with how critical the kernel is, then it would decides who gets the dram first

for eg: 256**3 gemm:
-the accel at 19.2gb/s peak dram, the gemm takes about 140 us while running alone, since it is dram bounded
-cpu at 4 cores: 172us and compute bounded, it use about half the dram peak (10.5gb/s)
=> nearly tied, but once the accel has to share dram with other kernels, then the cpu would win


what was added:

accel.py this one drives scalesim: it writes every gemm shape into one scalesim topology file, one layer per shape
-it calls scalesim().run_scale() which models the 32x32 array from the config file
-it also reads the scalesim csv files: compute report for cycles and detailed access report for dram words
-converts cycle to ns at 1ghz and words to bytes (x4 for fp32) and a bgemm row is b copies of one gemm

cpu_kernels.c is what gem5 runs: gem5 simulates this c script which is an arm program that runs the kernel
-calls the real openblas function for each kernel type + plain loops for elementwise and reduce
-put gem5 marker instr around the kernel call,
-compile inside docker

cpu.py which drives gem5, run gcc in the docker container to compile cpu_kernel.c, and for run, each shape and core count, it starts a container running gem5.opt gem5_se.py in spikes/ which that file would builds the simulated machine: n neoverse v2 cores, caches, ideal mem or 12 mib llc + ddr4
-simseconds gives the time, dram controller's bytesread and byteswritten give the traffic
-run 6 at once, cahce, check => run 6 containers in parallel

table.py is the kernel table where:
-distinct() reads every workload in .json format and return the 48 distinct pairs (type,shape) => to do list for two runners
-add() takes rows from a runner and merges them into kernel table.csv, replace any row with the same key at the same time
-load() and lookup() are how later code reads the table


=> what is left:

loss curve fomr the ramulator
simgrid backend
implementing the policies

outputs (going thru):

for the accel one:
gemm k = x, m = y, n = z => kernel type and shape: an m * k matrix times a k * n matrix
20.4us => compute time on the array, scalesim is set to an assumption that dram never stalls it so simgrid would add the dram time later
util 0.81 => fraction of the array's (1024 mult add per cycle) acctually used
2.56 mib => dram traffic: reads of both input matrices + the write of the output, 4 bytes per value
3.42x the minimum => traffic compared with the ideal of reading each input once and writing the output once

the one formula: cycles = ceil(m/32) * ceil(n/32) * (k + 62) for 32x32 tile at a time + 62 cycles of fill and drain
=> for eg check: 256**3: 8x8 x (256+62) = 20352 cycles => 20.4us
=> k = 3072, m = 128, n = 768: (128/32) x (768/32) x (3072 + 62) = 4 x 24 x 3134 = 300864 => 300.9us

=>deep k is efficient: a tile is useful for k of its k + 62 cycels so k=64 gives 64/126 = 0.51, and for a larger k like 3072 it would be 0.98
=>edge tile waste some of the array like m=196 need 7 tiles covering 224 rows, so 196/244 = 0.875 of them do work => mult w depth factor 0.94 at k=1024 and you get 0.83


for cpu one:
gemms scale very good with multicores like k=3072 m=128 n=768 goes from 11.4ms to 2.9ms
matrix vector gemms m =1 => scale about 2x, since theyre limited by meomry not compute


=>comparison with accels
Shape	            Compute: accel vs CPU 4 cores	Bytes: accel vs CPU	            Running alone at 19.2 GB/s
256³	            20.4 vs 172 µs (8.4x faster)	2.56 vs 1.73 MiB (1.5x more)	accel wins, 140 vs 172 µs
k=3072 m=128 n=768	301 vs 2902 µs (9.6x faster)	72.4 vs 11.9 MiB (6.1x more)	CPU wins, 2.9 vs 3.95 ms
bgemm k=64	        24.2 vs 173 µs (7.1x faster)	1.50 vs 2.32 MiB (less)	        accel wins, 82 vs 173 µs
m=1 k=2048 n=1000	67.5 vs 135 µs (2x faster)	    7.82 vs 7.84 MiB (same)	        tie, about 427 µs each


(.venv) hhlong@Longs-M3-Pro gchill-algorithm % python -m table.cpu
  results/gem5/table/bgemm_12_128_64_128_c1_ideal: 48 s
  results/gem5/table/bgemm_12_128_128_64_c1_ideal: 49 s
  results/gem5/table/bgemm_12_128_128_64_c2_ideal: 98 s
  results/gem5/table/bgemm_12_128_64_128_c2_ideal: 101 s
  results/gem5/table/elementwise_100352_c1_ideal: 23 s
  results/gem5/table/elementwise_196608_c1_ideal: 29 s
  results/gem5/table/elementwise_25088_c1_ideal: 6 s
  results/gem5/table/elementwise_200704_c1_ideal: 23 s
  results/gem5/table/bgemm_12_128_64_128_c3_ideal: 145 s
  results/gem5/table/elementwise_50176_c1_ideal: 8 s
  results/gem5/table/elementwise_768_c1_ideal: 4 s
  results/gem5/table/elementwise_393216_c1_ideal: 32 s
  results/gem5/table/bgemm_12_128_64_128_c4_ideal: 172 s
  results/gem5/table/bgemm_12_128_128_64_c3_ideal: 125 s
  results/gem5/table/elementwise_401408_c1_ideal: 30 s
  results/gem5/table/elementwise_98304_c1_ideal: 8 s
  results/gem5/table/bgemm_12_128_128_64_c4_ideal: 145 s
  results/gem5/table/elementwise_802816_c1_ideal: 52 s
  results/gem5/table/gemm_196_256_1024_c1_ideal: 67 s
  results/gem5/table/gemm_196_256_1024_c2_ideal: 108 s
  results/gem5/table/gemm_196_512_1024_c1_ideal: 116 s
  results/gem5/table/gemm_196_256_1024_c3_ideal: 151 s
  results/gem5/table/gemm_196_256_1024_c4_ideal: 184 s
  results/gem5/table/gemm_196_512_1024_c2_ideal: 184 s
  results/gem5/table/gemm_49_2048_1024_c1_ideal: 170 s
  results/gem5/table/gemm_196_512_1024_c3_ideal: 276 s
  results/gem5/table/gemm_196_512_1024_c4_ideal: 333 s
  results/gem5/table/gemm_784_128_1152_c1_ideal: 152 s
  results/gem5/table/gemm_49_2048_1024_c2_ideal: 333 s
  results/gem5/table/gemm_784_512_128_c1_ideal: 70 s
  results/gem5/table/gemm_784_128_1152_c2_ideal: 244 s
  results/gem5/table/gemm_784_512_128_c2_ideal: 108 s
  results/gem5/table/gemm_49_2048_1024_c3_ideal: 506 s
  results/gem5/table/gemm_784_512_128_c3_ideal: 146 s
  results/gem5/table/gemm_784_128_1152_c3_ideal: 319 s
  results/gem5/table/gemm_49_2048_1024_c4_ideal: 618 s
  results/gem5/table/gemm_12544_64_147_c1_ideal: 161 s
  results/gem5/table/gemm_784_512_128_c4_ideal: 197 s
  results/gem5/table/gemm_784_128_1152_c4_ideal: 402 s
  results/gem5/table/gemm_1_1000_2048_c1_ideal: 58 s
  results/gem5/table/gemm_1_1000_2048_c2_ideal: 166 s
  results/gem5/table/gemm_12544_64_147_c2_ideal: 340 s
  results/gem5/table/gemm_49_512_2048_c1_ideal: 86 s
  results/gem5/table/gemm_1_1000_2048_c3_ideal: 268 s
  results/gem5/table/gemm_12544_64_147_c3_ideal: 466 s
  results/gem5/table/gemm_49_512_2048_c2_ideal: 155 s
  results/gem5/table/gemm_1_1000_2048_c4_ideal: 382 s
  results/gem5/table/gemm_49_512_2048_c3_ideal: 221 s
  results/gem5/table/gemm_196_256_2304_c1_ideal: 132 s
  results/gem5/table/gemm_196_1024_256_c1_ideal: 58 s
  results/gem5/table/gemm_49_512_2048_c4_ideal: 292 s
  results/gem5/table/gemm_196_256_2304_c2_ideal: 211 s
  results/gem5/table/gemm_12544_64_147_c4_ideal: 625 s
  results/gem5/table/gemm_256_256_256_c1_ideal: 25 s
  results/gem5/table/gemm_196_1024_256_c2_ideal: 105 s
  results/gem5/table/gemm_256_256_256_c2_ideal: 39 s
  results/gem5/table/gemm_196_1024_256_c3_ideal: 140 s
  results/gem5/table/gemm_256_256_256_c3_ideal: 59 s
  results/gem5/table/gemm_256_256_256_c4_ideal: 71 s
  results/gem5/table/gemm_196_256_2304_c3_ideal: 313 s
  results/gem5/table/gemm_196_1024_256_c4_ideal: 174 s
  results/gem5/table/gemm_3136_64_256_c1_ideal: 65 s
  results/gem5/table/gemm_3136_128_256_c1_ideal: 117 s
  results/gem5/table/gemm_196_256_2304_c4_ideal: 372 s
  results/gem5/table/gemm_3136_128_256_c2_ideal: 203 s
  results/gem5/table/gemm_3136_64_256_c2_ideal: 131 s
  results/gem5/table/gemm_3136_64_256_c3_ideal: 189 s
  results/gem5/table/gemm_3136_128_256_c3_ideal: 274 s
  results/gem5/table/gemm_784_512_256_c1_ideal: 113 s
  results/gem5/table/gemm_3136_128_256_c4_ideal: 344 s
  results/gem5/table/gemm_3136_64_256_c4_ideal: 257 s
  results/gem5/table/gemm_784_512_256_c2_ideal: 166 s
  results/gem5/table/gemm_784_512_256_c3_ideal: 216 s
  results/gem5/table/gemm_784_512_256_c4_ideal: 262 s
  results/gem5/table/gemm_128_768_3072_c1_ideal: 322 s
  results/gem5/table/gemm_49_512_4608_c1_ideal: 173 s
  results/gem5/table/gemm_49_512_4608_c2_ideal: 324 s
  results/gem5/table/gemm_128_768_3072_c2_ideal: 516 s
  results/gem5/table/gemm_196_1024_512_c1_ideal: 109 s
  results/gem5/table/gemm_196_1024_512_c2_ideal: 171 s
  results/gem5/table/gemm_128_768_3072_c3_ideal: 721 s
  results/gem5/table/gemm_49_512_4608_c3_ideal: 498 s
  results/gem5/table/gemm_49_2048_512_c1_ideal: 85 s
  results/gem5/table/gemm_196_1024_512_c3_ideal: 254 s
  results/gem5/table/gemm_128_768_3072_c4_ideal: 877 s
  results/gem5/table/gemm_49_2048_512_c2_ideal: 168 s
  results/gem5/table/gemm_49_512_4608_c4_ideal: 674 s
  results/gem5/table/gemm_196_1024_512_c4_ideal: 319 s
  results/gem5/table/gemm_512_512_512_c1_ideal: 148 s
  results/gem5/table/gemm_49_2048_512_c3_ideal: 241 s
  results/gem5/table/gemm_784_128_512_c1_ideal: 67 s
  results/gem5/table/gemm_512_512_512_c2_ideal: 208 s
  results/gem5/table/gemm_784_128_512_c2_ideal: 108 s
  results/gem5/table/gemm_49_2048_512_c4_ideal: 393 s
  results/gem5/table/gemm_512_512_512_c3_ideal: 317 s
  results/gem5/table/gemm_784_128_512_c3_ideal: 183 s
  results/gem5/table/gemm_784_256_512_c1_ideal: 154 s
  results/gem5/table/gemm_512_512_512_c4_ideal: 383 s
  results/gem5/table/gemm_784_128_512_c4_ideal: 269 s
  results/gem5/table/gemm_784_256_512_c2_ideal: 195 s
  results/gem5/table/gemm_3136_64_576_c1_ideal: 186 s
  results/gem5/table/gemm_784_256_512_c3_ideal: 261 s
  results/gem5/table/gemm_3136_256_64_c1_ideal: 80 s
  results/gem5/table/gemm_784_256_512_c4_ideal: 325 s
  results/gem5/table/gemm_3136_256_64_c2_ideal: 168 s
  results/gem5/table/gemm_3136_64_576_c2_ideal: 347 s
  results/gem5/table/gemm_3136_64_64_c1_ideal: 29 s
  results/gem5/table/gemm_3136_64_64_c2_ideal: 60 s
  results/gem5/table/gemm_3136_256_64_c3_ideal: 226 s
  results/gem5/table/gemm_3136_64_64_c3_ideal: 98 s
  results/gem5/table/gemm_3136_64_576_c3_ideal: 473 s
  results/gem5/table/gemm_3136_64_64_c4_ideal: 126 s
  results/gem5/table/gemm_3136_256_64_c4_ideal: 319 s
  results/gem5/table/gemm_128_768_768_c1_ideal: 103 s
  results/gem5/table/gemm_3136_64_576_c4_ideal: 661 s
  results/gem5/table/gemm_128_768_768_c2_ideal: 157 s
  results/gem5/table/gemm_128_3072_768_c1_ideal: 405 s
  results/gem5/table/gemm_1_768_768_c1_ideal: 20 s
  results/gem5/table/gemm_1_768_768_c2_ideal: 67 s
  results/gem5/table/gemm_128_768_768_c3_ideal: 257 s
  results/gem5/table/gemm_1_768_768_c3_ideal: 110 s
  results/gem5/table/gemm_128_3072_768_c2_ideal: 632 s
  results/gem5/table/gemm_128_768_768_c4_ideal: 293 s
  results/gem5/table/getrf_512_c1_ideal: 111 s
  results/gem5/table/gemm_1_768_768_c4_ideal: 151 s
  results/gem5/table/potrf_256_c1_ideal: 15 s
  results/gem5/table/potrf_256_c2_ideal: 24 s
  results/gem5/table/potrf_256_c3_ideal: 43 s
  results/gem5/table/potrf_256_c4_ideal: 56 s
  results/gem5/table/potrf_512_c1_ideal: 49 s
  results/gem5/table/gemm_128_3072_768_c3_ideal: 852 s
  results/gem5/table/getrf_512_c2_ideal: 279 s
  results/gem5/table/potrf_512_c2_ideal: 99 s
  results/gem5/table/reduce_100352_c1_ideal: 9 s
  results/gem5/table/reduce_802816_c1_ideal: 42 s
  results/gem5/table/getrf_512_c3_ideal: 376 s
  results/gem5/table/potrf_512_c3_ideal: 142 s
  results/gem5/table/syrk_256_256_c1_ideal: 18 s
  results/gem5/table/syrk_256_256_c2_ideal: 30 s
  results/gem5/table/getrf_512_c4_ideal: 381 s
  results/gem5/table/syrk_256_256_c3_ideal: 42 s
  results/gem5/table/gemm_128_3072_768_c4_ideal: 1019 s
  results/gem5/table/syrk_256_256_c4_ideal: 59 s
  results/gem5/table/potrf_512_c4_ideal: 187 s
  results/gem5/table/trsm_256_256_c1_ideal: 21 s
  results/gem5/table/trsm_256_256_c2_ideal: 35 s
  results/gem5/table/syrk_512_512_c1_ideal: 91 s
  results/gem5/table/trsm_256_256_c3_ideal: 44 s
  results/gem5/table/syrk_512_512_c2_ideal: 132 s
  results/gem5/table/trsm_256_256_c4_ideal: 53 s
  results/gem5/table/syrk_512_512_c3_ideal: 164 s
  results/gem5/table/trsm_512_512_c1_ideal: 93 s
  results/gem5/table/trsm_512_512_c2_ideal: 124 s
  results/gem5/table/syrk_512_512_c4_ideal: 195 s
  results/gem5/table/bgemm_12_128_64_128_c1_ddr: 108 s
  results/gem5/table/trsm_512_512_c3_ideal: 154 s
  results/gem5/table/bgemm_12_128_128_64_c1_ddr: 109 s
  results/gem5/table/elementwise_100352_c1_ddr: 95 s
  results/gem5/table/trsm_512_512_c4_ideal: 190 s
  results/gem5/table/elementwise_196608_c1_ddr: 102 s
  results/gem5/table/elementwise_25088_c1_ddr: 91 s
  results/gem5/table/elementwise_200704_c1_ddr: 103 s
  results/gem5/table/elementwise_393216_c1_ddr: 112 s
  results/gem5/table/elementwise_50176_c1_ddr: 94 s
  results/gem5/table/elementwise_768_c1_ddr: 89 s
  results/gem5/table/elementwise_401408_c1_ddr: 112 s
  results/gem5/table/elementwise_98304_c1_ddr: 96 s
  results/gem5/table/elementwise_802816_c1_ddr: 135 s
  results/gem5/table/gemm_196_256_1024_c1_ddr: 139 s
  results/gem5/table/gemm_196_512_1024_c1_ddr: 181 s
  results/gem5/table/gemm_784_512_128_c1_ddr: 133 s
  results/gem5/table/gemm_784_128_1152_c1_ddr: 198 s
  results/gem5/table/gemm_49_2048_1024_c1_ddr: 218 s
  results/gem5/table/gemm_1_1000_2048_c1_ddr: 129 s
  results/gem5/table/gemm_12544_64_147_c1_ddr: 219 s
  results/gem5/table/gemm_256_256_256_c1_ddr: 100 s
  results/gem5/table/gemm_49_512_2048_c1_ddr: 150 s
  results/gem5/table/gemm_196_1024_256_c1_ddr: 132 s
  results/gem5/table/gemm_196_256_2304_c1_ddr: 191 s
  results/gem5/table/gemm_3136_128_256_c1_ddr: 185 s
  results/gem5/table/gemm_3136_64_256_c1_ddr: 143 s
  results/gem5/table/gemm_784_512_256_c1_ddr: 172 s
  results/gem5/table/gemm_196_1024_512_c1_ddr: 173 s
  results/gem5/table/gemm_49_512_4608_c1_ddr: 230 s
  results/gem5/table/gemm_49_2048_512_c1_ddr: 147 s
  results/gem5/table/gemm_784_128_512_c1_ddr: 132 s
  results/gem5/table/gemm_512_512_512_c1_ddr: 198 s
  results/gem5/table/gemm_128_768_3072_c1_ddr: 360 s
  results/gem5/table/gemm_3136_64_64_c1_ddr: 112 s
  results/gem5/table/gemm_3136_256_64_c1_ddr: 152 s
  results/gem5/table/gemm_784_256_512_c1_ddr: 187 s
  results/gem5/table/gemm_3136_64_576_c1_ddr: 223 s
  results/gem5/table/gemm_1_768_768_c1_ddr: 98 s
  results/gem5/table/potrf_256_c1_ddr: 91 s
  results/gem5/table/gemm_128_768_768_c1_ddr: 164 s
  results/gem5/table/getrf_512_c1_ddr: 164 s
  results/gem5/table/potrf_512_c1_ddr: 122 s
  results/gem5/table/reduce_100352_c1_ddr: 89 s
  results/gem5/table/syrk_256_256_c1_ddr: 93 s
  results/gem5/table/reduce_802816_c1_ddr: 118 s
  results/gem5/table/trsm_256_256_c1_ddr: 85 s
  results/gem5/table/gemm_128_3072_768_c1_ddr: 362 s
  results/gem5/table/syrk_512_512_c1_ddr: 130 s
  results/gem5/table/trsm_512_512_c1_ddr: 132 s
bgemm       b=12;k=128;m=128;n=64  us at 1..4 cores:    514.0     276.0     214.0     166.0     1.96 MiB (1.30x the minimum)
bgemm       b=12;k=64;m=128;n=128  us at 1..4 cores:    526.0     285.0     295.0     173.0     2.32 MiB (1.55x the minimum)
elementwise n=100352               us at 1..1 cores:     67.0     1.15 MiB (1.50x the minimum)
elementwise n=196608               us at 1..1 cores:    131.0     2.25 MiB (1.50x the minimum)
elementwise n=200704               us at 1..1 cores:    134.0     2.30 MiB (1.50x the minimum)
elementwise n=25088                us at 1..1 cores:     17.0     0.29 MiB (1.53x the minimum)
elementwise n=393216               us at 1..1 cores:    268.0     4.50 MiB (1.50x the minimum)
elementwise n=401408               us at 1..1 cores:    274.0     4.60 MiB (1.50x the minimum)
elementwise n=50176                us at 1..1 cores:     34.0     0.58 MiB (1.51x the minimum)
elementwise n=768                  us at 1..1 cores:      1.0     0.01 MiB (1.97x the minimum)
elementwise n=802816               us at 1..1 cores:    551.0     9.19 MiB (1.50x the minimum)
elementwise n=98304                us at 1..1 cores:     66.0     1.13 MiB (1.50x the minimum)
gemm        k=1024;m=196;n=256     us at 1..4 cores:   1953.0    1047.0     792.0     551.0     3.00 MiB (1.53x the minimum)
gemm        k=1024;m=196;n=512     us at 1..4 cores:   3856.0    2046.0    1498.0    1047.0     4.38 MiB (1.39x the minimum)
gemm        k=1024;m=49;n=2048     us at 1..4 cores:   4297.0    2194.0    1519.0    1130.0     9.43 MiB (1.10x the minimum)
gemm        k=1152;m=784;n=128     us at 1..4 cores:   4418.0    2236.0    1535.0    1154.0     5.14 MiB (1.17x the minimum)
gemm        k=128;m=784;n=512      us at 1..4 cores:   1954.0     985.0     676.0     507.0     4.54 MiB (2.10x the minimum)
gemm        k=147;m=12544;n=64     us at 1..4 cores:   4698.0    2402.0    1602.0    1219.0    13.29 MiB (1.31x the minimum)
gemm        k=2048;m=1;n=1000      us at 1..4 cores:    251.0     186.0     149.0     135.0     7.84 MiB (1.00x the minimum)
gemm        k=2048;m=49;n=512      us at 1..4 cores:   2161.0    1108.0     799.0     582.0     5.05 MiB (1.13x the minimum)
gemm        k=2304;m=196;n=256     us at 1..4 cores:   4392.0    2350.0    1788.0    1224.0     5.20 MiB (1.25x the minimum)
gemm        k=256;m=196;n=1024     us at 1..4 cores:   1929.0    1009.0     708.0     517.0     3.34 MiB (1.70x the minimum)
gemm        k=256;m=256;n=256      us at 1..4 cores:    636.0     323.0     265.0     172.0     1.73 MiB (2.31x the minimum)
gemm        k=256;m=3136;n=128     us at 1..4 cores:   3881.0    1959.0    1321.0     991.0     6.52 MiB (1.38x the minimum)
gemm        k=256;m=3136;n=64      us at 1..4 cores:   2016.0    1035.0     689.0     520.0     4.81 MiB (1.24x the minimum)
gemm        k=256;m=784;n=512      us at 1..4 cores:   3804.0    1917.0    1320.0     982.0     6.03 MiB (2.16x the minimum)
gemm        k=3072;m=128;n=768     us at 1..4 cores:  11409.0    5722.0    4199.0    2902.0    11.92 MiB (1.10x the minimum)
gemm        k=4608;m=49;n=512      us at 1..4 cores:   4868.0    2493.0    1790.0    1287.0    10.52 MiB (1.06x the minimum)
gemm        k=512;m=196;n=1024     us at 1..4 cores:   3848.0    2018.0    1426.0    1028.0     4.53 MiB (1.44x the minimum)
gemm        k=512;m=49;n=2048      us at 1..4 cores:   2149.0    1089.0     756.0     571.0     5.21 MiB (1.16x the minimum)
gemm        k=512;m=512;n=512      us at 1..4 cores:   4969.0    2493.0    1859.0    1268.0     5.21 MiB (1.74x the minimum)
gemm        k=512;m=784;n=128      us at 1..4 cores:   1954.0     991.0     681.0     515.0     2.82 MiB (1.30x the minimum)
gemm        k=512;m=784;n=256      us at 1..4 cores:   3839.0    1934.0    1332.0    1006.0     5.28 MiB (1.89x the minimum)
gemm        k=576;m=3136;n=64      us at 1..4 cores:   4537.0    2304.0    1550.0    1165.0     8.74 MiB (1.12x the minimum)
gemm        k=64;m=3136;n=256      us at 1..4 cores:   2093.0    1034.0     693.0     526.0     8.49 MiB (2.18x the minimum)
gemm        k=64;m=3136;n=64       us at 1..4 cores:    537.0     275.0     186.0     141.0     2.36 MiB (1.52x the minimum)
gemm        k=768;m=128;n=3072     us at 1..4 cores:  11402.0    5720.0    4158.0    2895.0    13.03 MiB (1.20x the minimum)
gemm        k=768;m=128;n=768      us at 1..4 cores:   2872.0    1442.0    1060.0     736.0     4.03 MiB (1.34x the minimum)
gemm        k=768;m=1;n=768        us at 1..4 cores:     68.0      43.0      34.0      31.0     2.27 MiB (1.01x the minimum)
getrf       b=512                  us at 1..4 cores:   6552.0    5506.0    4966.0    3516.0     5.10 MiB (2.55x the minimum)
potrf       b=256                  us at 1..4 cores:    360.0     307.0     304.0     305.0     0.86 MiB (1.71x the minimum)
potrf       b=512                  us at 1..4 cores:   1817.0    1372.0    1293.0    1142.0     3.00 MiB (1.50x the minimum)
reduce      n=100352               us at 1..1 cores:     67.0     0.39 MiB (1.01x the minimum)
reduce      n=802816               us at 1..1 cores:    535.0     3.07 MiB (1.00x the minimum)
syrk        k=256;n=256            us at 1..4 cores:    356.0     215.0     162.0     146.0     1.25 MiB (1.67x the minimum)
syrk        k=512;n=512            us at 1..4 cores:   2641.0    1486.0    1002.0     760.0     3.42 MiB (1.14x the minimum)
trsm        m=256;n=256            us at 1..4 cores:    511.0     288.0     239.0     149.0     1.20 MiB (1.60x the minimum)
trsm        m=512;n=512            us at 1..4 cores:   3217.0    1643.0    1276.0     890.0     4.02 MiB (1.34x the minimum)
156 cpu rows -> data/kernel_table.csv




(.venv) hhlong@Longs-M3-Pro gchill-algorithm % python -m table.accel
bgemm b=12;k=128;m=128;n=64       18.2 us  util 0.67     1.50 MiB  (1.00x the minimum)
bgemm b=12;k=64;m=128;n=128       24.2 us  util 0.51     1.50 MiB  (1.00x the minimum)
gemm  k=1024;m=196;n=256          60.8 us  util 0.83    13.32 MiB  (6.80x the minimum)
gemm  k=1024;m=196;n=512         121.6 us  util 0.83    26.63 MiB  (8.46x the minimum)
gemm  k=1024;m=49;n=2048         139.0 us  util 0.72    28.63 MiB  (3.34x the minimum)
gemm  k=1152;m=784;n=128         121.4 us  util 0.93    28.23 MiB  (6.43x the minimum)
gemm  k=128;m=784;n=512           76.0 us  util 0.66     7.94 MiB  (3.67x the minimum)
gemm  k=147;m=12544;n=64         163.9 us  util 0.70    17.17 MiB  (1.69x the minimum)
gemm  k=2048;m=1;n=1000           67.5 us  util 0.03     7.82 MiB  (1.00x the minimum)
gemm  k=2048;m=49;n=512           67.5 us  util 0.74    14.22 MiB  (3.18x the minimum)
gemm  k=2304;m=196;n=256         132.5 us  util 0.85    29.72 MiB  (7.14x the minimum)
gemm  k=256;m=196;n=1024          71.2 us  util 0.70     8.20 MiB  (4.19x the minimum)
gemm  k=256;m=256;n=256           20.4 us  util 0.81     2.56 MiB  (3.42x the minimum)
gemm  k=256;m=3136;n=128         124.7 us  util 0.81    13.94 MiB  (2.95x the minimum)
gemm  k=256;m=3136;n=64           62.3 us  util 0.81     6.95 MiB  (1.79x the minimum)
gemm  k=256;m=784;n=512          127.2 us  util 0.79    14.44 MiB  (5.16x the minimum)
gemm  k=3072;m=128;n=768         300.9 us  util 0.98    72.38 MiB  (6.66x the minimum)
gemm  k=4608;m=49;n=512          149.4 us  util 0.76    31.88 MiB  (3.20x the minimum)
gemm  k=512;m=196;n=1024         128.6 us  util 0.78    16.95 MiB  (5.38x the minimum)
gemm  k=512;m=49;n=2048           73.5 us  util 0.68     8.41 MiB  (1.88x the minimum)
gemm  k=512;m=512;n=512          146.9 us  util 0.89    18.94 MiB  (6.31x the minimum)
gemm  k=512;m=784;n=128           57.4 us  util 0.87     6.95 MiB  (3.21x the minimum)
gemm  k=512;m=784;n=256          114.8 us  util 0.87    13.95 MiB  (4.99x the minimum)
gemm  k=576;m=3136;n=64          125.0 us  util 0.90    14.74 MiB  (1.89x the minimum)
gemm  k=64;m=3136;n=256           98.8 us  util 0.51     9.25 MiB  (2.38x the minimum)
gemm  k=64;m=3136;n=64            24.7 us  util 0.51     2.31 MiB  (1.50x the minimum)
gemm  k=768;m=128;n=3072         318.7 us  util 0.93    49.46 MiB  (4.55x the minimum)
gemm  k=768;m=128;n=768           79.7 us  util 0.93    12.34 MiB  (4.11x the minimum)
gemm  k=768;m=1;n=768             19.9 us  util 0.03     2.26 MiB  (1.00x the minimum)
29 accel rows -> data/kernel_table.csv