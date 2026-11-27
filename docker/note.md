these are for running inside of colima (replace for docker)

dockerfile in chameloen makes the .rec files:

it would start from ubuntu and install starpu fxt which is the tracer and openblas from apt, then it builds chameleon from the source with fp32.

the trace.sh file would do these:

run chameleon_stesting -o s<op> with tracing on, for eg like potrf 4096 512 is 4096 wide matrix in 512 tiles, so 8x8 tiles
convert the tace with starpu fxt 
then writes tasks.rec thru stdout

=> pipleine becomes: trace.sh -> .rec -> chameleon.py -> json

dockerfile in gem5 is the cpu kernels for gem5 to run:

the image starts from gem5 build image and then adds openblas to it
gem5 is not in this image, it is in tools folder then it gets mounted in here

=> the file gem5_gemm.py use this image twice, one for compiling gem5_gemm.c against the openblas and run gem5 on the resulting binary
which the output would goes to result folder

=> this is how we will get cpu times for each kernel shape in data folder, workloads subforlder for the kernel table