# gem5's own build image + a static openblas for running kernels in gem5 se mode
# - TARGET=NEOVERSEV2 with GCCVERSIONGTEQ13=0: openblas's flags for older compilers (armv8.2-a+sve), which
#   keep the sve kernels; gcc 13's -mcpu=neoverse-v2 emits stlur stores that gem5 25.1 cannot decode
# - NO_AFFINITY=1 and mbind made a no-op: gem5 se mode has no mbind syscall (it only sets numa placement)
# - CROSS=1 skips openblas's tests: the mac's cores have no sve, so the neoverse v2 kernels cannot run here
FROM ghcr.io/gem5/ubuntu-24.04_all-dependencies:v25-1
RUN git clone --depth 1 --branch v0.3.34 https://github.com/OpenMathLib/OpenBLAS.git /src/openblas \
    && sed -i 's/return syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);/return 0;/' \
         /src/openblas/common_linux.h \
    && make -C /src/openblas -j 6 TARGET=NEOVERSEV2 GCCVERSIONGTEQ13=0 USE_OPENMP=0 NUM_THREADS=16 \
         NO_AFFINITY=1 NO_SHARED=1 CROSS=1 > /dev/null \
    && make -C /src/openblas PREFIX=/opt/openblas NO_SHARED=1 install > /dev/null