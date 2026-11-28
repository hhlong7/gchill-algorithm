//this is the script for all 8 kernel types

// one cpu kernel of the kernel table (B2), run in gem5 se mode by table/cpu.py
// usage: cpu_kernels <ddr 0|1> <type> <dims...>     threads come from OPENBLAS_NUM_THREADS
//   gemm m n k | bgemm b m n k | potrf b | getrf b | trsm m n | syrk n k | elementwise n | reduce n
// - row major fp32, openblas cblas / lapacke; getrf = lapacke sgetrf (openblas has no getrf_nopiv): the matrix
//   is diagonally dominant, so partial pivoting never swaps a row and the work is the same as nopiv
// - elementwise (y = 0.5 x + 1) and reduce (sum) are plain loops on one thread
// ddr 1 (the dram bytes run): scrub the caches first, so inputs start in dram, and after, so the dirty lines it
//   left are written back. gem5 stats blocks: 1 = the kernel, 2 = that write-back
// ddr 0 (the ideal-memory time runs): no scrubs, inputs start warm. the scrub is slow in gem5 (idle openblas threads
//   spin through it) and with zero-latency memory it changes the time by under 1% (gemm 256^3: 635 vs 637 us)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>

// gem5 pseudo instructions (util/m5/src/abi/arm64/m5op.S): 0xff000110 | func << 16, args in x0 x1
#define M5OP(func) __asm__ volatile("mov x0, #0\n\tmov x1, #0\n\t.inst 0xff000110 | (" #func " << 16)" \
                                    ::: "x0", "x1", "memory")
#define M5_RESET_STATS() M5OP(0x40)
#define M5_DUMP_RESET_STATS() M5OP(0x42)
#define M5_DUMP_STATS() M5OP(0x41)

// push everything out of the l1, l2 and llc by reading a clean buffer much bigger than all of them, twice
// (the l1 and l2 use rrip, which keeps old lines through one scan). gem5 se mode traps dc civac
static volatile char scrub_buf[32 << 20]; // .bss: zero pages, never written, so reading leaves no dirty lines
static void scrub(void) {
    volatile long sum = 0;
    for (int pass = 0; pass < 2; pass++)
        for (long i = 0; i < (long)sizeof scrub_buf; i += 64) sum += scrub_buf[i];
}

static float *matrix(long rows, long cols, int seed) {
    float *a = malloc(sizeof(float) * rows * cols);
    for (long i = 0; i < rows * cols; i++) a[i] = 1.0f / (i % seed + 1);
    return a;
}

// symmetric, diagonally dominant b x b (so potrf succeeds and getrf never pivots)
static float *dominant(int b) {
    float *a = malloc(sizeof(float) * b * b);
    for (int i = 0; i < b; i++)
        for (int j = 0; j < b; j++) a[(long)i * b + j] = i == j ? (float)b : 1.0f / (abs(i - j) + 1);
    return a;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: cpu_kernels <ddr 0|1> <type> <dims...>\n"); return 1; }
    int ddr = atoi(argv[1]);
    const char *t = argv[2];
    long d[4] = {0};
    for (int i = 3; i < argc && i < 7; i++) d[i - 3] = atol(argv[i]);
    float *a = 0, *b = 0, *c = 0;
    int *ipiv = 0;
    if (!strcmp(t, "gemm")) { a = matrix(d[0], d[2], 7); b = matrix(d[2], d[1], 5); c = matrix(d[0], d[1], 3); }
    else if (!strcmp(t, "bgemm")) { a = matrix(d[0] * d[1], d[3], 7); b = matrix(d[0] * d[3], d[2], 5);
                                    c = matrix(d[0] * d[1], d[2], 3); }
    else if (!strcmp(t, "potrf") || !strcmp(t, "getrf")) { a = dominant(d[0]); ipiv = malloc(sizeof(int) * d[0]); }
    else if (!strcmp(t, "trsm")) { a = dominant(d[0]); b = matrix(d[0], d[1], 5); }
    else if (!strcmp(t, "syrk")) { a = matrix(d[0], d[1], 7); c = matrix(d[0], d[0], 3); }
    else if (!strcmp(t, "elementwise") || !strcmp(t, "reduce")) { a = matrix(d[0], 1, 7); c = matrix(d[0], 1, 3); }
    else { fprintf(stderr, "unknown kernel type %s\n", t); return 1; }
    float w[64] = {0};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 8, 8, 8, 1.0f, w, 8, w, 8, 0.0f, w, 8); // start openblas threads
    if (ddr) scrub(); // inputs start in dram, like a kernel whose producer ran long ago
    float sum = 0;
    M5_RESET_STATS();
    if (!strcmp(t, "gemm"))
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, d[0], d[1], d[2], 1.0f, a, d[2], b, d[1], 0.0f, c, d[1]);
    else if (!strcmp(t, "bgemm"))
        for (long i = 0; i < d[0]; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, d[1], d[2], d[3], 1.0f, a + i * d[1] * d[3], d[3],
                        b + i * d[3] * d[2], d[2], 0.0f, c + i * d[1] * d[2], d[2]);
    else if (!strcmp(t, "potrf")) sum = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', d[0], a, d[0]);
    else if (!strcmp(t, "getrf")) sum = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, d[0], d[0], a, d[0], ipiv);
    else if (!strcmp(t, "trsm"))
        cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, d[0], d[1], 1.0f, a, d[0], b, d[1]);
    else if (!strcmp(t, "syrk"))
        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, d[0], d[1], -1.0f, a, d[1], 1.0f, c, d[0]);
    else if (!strcmp(t, "elementwise")) for (long i = 0; i < d[0]; i++) c[i] = 0.5f * a[i] + 1.0f;
    else for (long i = 0; i < d[0]; i++) sum += a[i];
    if (ddr) {
        M5_DUMP_RESET_STATS();
        scrub(); // write back the output (and anything else) it left dirty in the caches
    }
    M5_DUMP_STATS();
    if (sum != 0 && strcmp(t, "reduce")) { fprintf(stderr, "%s failed: info %f\n", t, sum); return 1; }
    printf("%s ok %f\n", t, sum + (a ? a[0] : 0) + (c ? c[0] : 0));
    return 0;
}