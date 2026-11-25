// one openblas sgemm (row major, C = A B with A m x k, B k x n) measured in gem5 from cold caches
// usage: gemm m n k    threads come from OPENBLAS_NUM_THREADS
// gem5 stats blocks: 1 = the gemm (time, dram reads and writes), 2 = writing back the dirty lines it left
// (dram bytes are only measured on 1 core: the scrub runs on the main thread)
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

// gem5 pseudo instructions (util/m5/src/abi/arm64/m5op.S): 0xff000110 | func << 16, args in x0 x1
#define M5OP(func) __asm__ volatile("mov x0, #0\n\tmov x1, #0\n\t.inst 0xff000110 | (" #func " << 16)" \
                                    ::: "x0", "x1", "memory")
#define M5_RESET_STATS() M5OP(0x40)
#define M5_DUMP_STATS() M5OP(0x41)
#define M5_DUMP_RESET_STATS() M5OP(0x42)

// push everything out of this core's l1, l2 and the llc by reading a clean buffer much bigger than all of
// them (twice: the l1 and l2 use rrip, which keeps old lines through one scan). gem5 se mode traps dc civac
static volatile char scrub_buf[32 << 20]; // in .bss: zero pages, never written, so reading it leaves no dirty lines
static void scrub(void) {
    volatile long sum = 0;
    for (int pass = 0; pass < 2; pass++)
        for (long i = 0; i < (long)sizeof scrub_buf; i += 64) sum += scrub_buf[i];
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: gemm m n k\n"); return 1; }
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    long sa = sizeof(float) * m * k, sb = sizeof(float) * k * n, sc = sizeof(float) * m * n;
    float *a = malloc(sa), *b = malloc(sb), *c = malloc(sc);
    for (long i = 0; i < (long)m * k; i++) a[i] = 1.0f / (i % 7 + 1);
    for (long i = 0; i < (long)k * n; i++) b[i] = 1.0f / (i % 5 + 1);
    for (long i = 0; i < (long)m * n; i++) c[i] = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 8, 8, 8, 1.0f, a, k, b, n, 0.0f, c, n); // start openblas threads
    scrub(); // inputs start in dram, like a kernel whose producer ran long ago
    M5_RESET_STATS();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
    M5_DUMP_RESET_STATS();
    scrub(); // write back the output (and anything else) it left dirty in the caches
    M5_DUMP_STATS();
    printf("gemm %d %d %d c[0]=%f\n", m, n, k, c[0]);
    return 0;
}