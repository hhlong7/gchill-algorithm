this is the raw traces. the input

the files have a list of records separated by blank lines. 
they are what starpu recorded while chameleon really ran a factorization

Kernel records look like this:
Name: spotrf = which BLAS/LAPACK kernel (s = fp32)
Model: spotrf = present only on real kernels
JobId: 109
SubmitOrder: 1 = order Chameleon submitted it in; we number kernels by this
DependsOn: 38 = jobs it waits for (38 is a "ghost": outside the trace, so it gets dropped)
Parameters: M512x512x6 = tile shape, gives the 512x512 tile
GFlop / StartTime / EndTime 

=> counts per file:
chol8x512 = 387 records and 120 real kernels
lu8x512 = 772 records and 204 real kernels
chol16x256 = 1859 records and 816 real kernles
