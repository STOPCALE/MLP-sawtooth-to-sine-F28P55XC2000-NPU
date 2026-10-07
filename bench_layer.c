/* Benchmark tool (NOT teaching material): A/B comparison of M-inline vs M-lookup.
 *
 * Question : how much does precomputing M in the "load stage" actually save?
 * Method   : two functions differing in exactly ONE line (where M comes from),
 *            timed in the same run with identical data.
 * Guards   : (1) varying input -> the call cannot be hoisted out of the loop
 *            (2) checksum over ALL outputs -> the call cannot be dead-code eliminated
 *            (3) 1e6 reps -> clock() resolution (1 ms on Windows) is amortised to <1%
 * Extra    : the two checksums must be identical -> proves only M's origin changed.
 *
 * Build: gcc -O2 -Wall bench_layer.c -o bench_layer.exe
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "quant_params.h"

#define REPS 1000000

/* runtime state produced by the "load stage" */
typedef struct { double M[64]; } layer_rt_t;

static void prepare(const qlayer_t *L, layer_rt_t *rt)
{
    for (int o = 0; o < L->out_dim; o++)
        rt->M[o] = L->S_in * (double)L->S_w[o] / L->S_out;
}

/* -------------------- v1: M computed inline (original) -------------------- */
static void layer_v1(const qlayer_t *L, const int8_t *x, int8_t *q_out)
{
    for (int o = 0; o < L->out_dim; o++) {
        int32_t acc = L->b_int[o];
        for (int i = 0; i < L->in_dim; i++)
            acc += ((int32_t)x[i] - L->z_in) * (int32_t)L->W_q[o * L->in_dim + i];
        double M = L->S_in * (double)L->S_w[o] / L->S_out;   /* <-- the only difference */
        int32_t r = (int32_t)nearbyint(acc * M) + L->z_out;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        q_out[o] = (int8_t)r;
    }
}

/* -------------------- v2: M looked up (precomputed) ---------------------- */
static void layer_v2(const qlayer_t *L, const layer_rt_t *rt,
                     const int8_t *x, int8_t *q_out)
{
    for (int o = 0; o < L->out_dim; o++) {
        int32_t acc = L->b_int[o];
        for (int i = 0; i < L->in_dim; i++)
            acc += ((int32_t)x[i] - L->z_in) * (int32_t)L->W_q[o * L->in_dim + i];
        double M = rt->M[o];                                 /* <-- the only difference */
        int32_t r = (int32_t)nearbyint(acc * M) + L->z_out;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        q_out[o] = (int8_t)r;
    }
}

/* ---------------- v3: precomputed M + manual rounding (no libm call) ------ */
/* Magic-number trick: 1.5*2^52. Adding it forces the mantissa to align at the
 * integer bit, and the FPU's DEFAULT rounding mode (nearest-even) does the job;
 * subtracting it back leaves the rounded integer. gcc emits addsd/subsd/cvttsd2si
 * -- three instructions instead of a call into libm. Valid for |x| < 2^51. */
static inline int32_t round_nearest_even(double x)
{
    const double MAGIC = 6755399441055744.0;   /* 1.5 * 2^52 */
    return (int32_t)((x + MAGIC) - MAGIC);
}

static void layer_v3(const qlayer_t *L, const layer_rt_t *rt,
                     const int8_t *x, int8_t *q_out)
{
    for (int o = 0; o < L->out_dim; o++) {
        int32_t acc = L->b_int[o];
        for (int i = 0; i < L->in_dim; i++)
            acc += ((int32_t)x[i] - L->z_in) * (int32_t)L->W_q[o * L->in_dim + i];
        int32_t r = round_nearest_even(acc * rt->M[o]) + L->z_out;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        q_out[o] = (int8_t)r;
    }
}

/* ---------------- v4: v3 + z_in folded into bias ---------------- */
static void layer_v4(const qlayer_t *L, const layer_rt_t *rt,
                     const int8_t *x, int8_t *q_out)
{
    for (int o = 0; o < L->out_dim; o++) {
        int32_t acc = L->b_fold[o];                                        /* folded bias */
        for (int i = 0; i < L->in_dim; i++)
            acc += (int32_t)x[i] * (int32_t)L->W_q[o * L->in_dim + i];     /* no (x - z_in) */
        int32_t r = round_nearest_even(acc * rt->M[o]) + L->z_out;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        q_out[o] = (int8_t)r;
    }
}

static double bench_v1(const qlayer_t *L, long long *cks)
{
    int8_t x[64], y[64];
    long long c = 0;
    for (int i = 0; i < 64; i++) x[i] = (int8_t)(i - 32);
    clock_t t0 = clock();
    for (int rep = 0; rep < REPS; rep++) {
        x[0] = (int8_t)(rep & 0x7f);          /* guard 1: input varies */
        layer_v1(L, x, y);
        for (int i = 0; i < L->out_dim; i++) c += y[i];   /* guard 2: output used */
    }
    clock_t t1 = clock();
    *cks = c;
    return (double)(t1 - t0) * 1e6 / CLOCKS_PER_SEC / REPS;
}

static double bench_v2(const qlayer_t *L, const layer_rt_t *rt, long long *cks)
{
    int8_t x[64], y[64];
    long long c = 0;
    for (int i = 0; i < 64; i++) x[i] = (int8_t)(i - 32);
    clock_t t0 = clock();
    for (int rep = 0; rep < REPS; rep++) {
        x[0] = (int8_t)(rep & 0x7f);
        layer_v2(L, rt, x, y);
        for (int i = 0; i < L->out_dim; i++) c += y[i];
    }
    clock_t t1 = clock();
    *cks = c;
    return (double)(t1 - t0) * 1e6 / CLOCKS_PER_SEC / REPS;
}

static double bench_v3(const qlayer_t *L, const layer_rt_t *rt, long long *cks)
{
    int8_t x[64], y[64];
    long long c = 0;
    for (int i = 0; i < 64; i++) x[i] = (int8_t)(i - 32);
    clock_t t0 = clock();
    for (int rep = 0; rep < REPS; rep++) {
        x[0] = (int8_t)(rep & 0x7f);
        layer_v3(L, rt, x, y);
        for (int i = 0; i < L->out_dim; i++) c += y[i];
    }
    clock_t t1 = clock();
    *cks = c;
    return (double)(t1 - t0) * 1e6 / CLOCKS_PER_SEC / REPS;
}

static double bench_v4(const qlayer_t *L, const layer_rt_t *rt, long long *cks)
{
    int8_t x[64], y[64];
    long long c = 0;
    for (int i = 0; i < 64; i++) x[i] = (int8_t)(i - 32);
    clock_t t0 = clock();
    for (int rep = 0; rep < REPS; rep++) {
        x[0] = (int8_t)(rep & 0x7f);
        layer_v4(L, rt, x, y);
        for (int i = 0; i < L->out_dim; i++) c += y[i];
    }
    clock_t t1 = clock();
    *cks = c;
    return (double)(t1 - t0) * 1e6 / CLOCKS_PER_SEC / REPS;
}

int main(void)
{
    layer_rt_t rt[3];
    for (int l = 0; l < 3; l++) prepare(&LAYERS[l], &rt[l]);

    printf("REPS = %d | clock() resolution = %d Hz\n", REPS, (int)CLOCKS_PER_SEC);
    printf("--------------------------------------------------------------------------------------\n");
    printf("%-21s %9s %9s %9s %9s     %-9s %s\n",
           "layer", "v1(us)", "v2(us)", "v3(us)", "v4(us)", "v1->v4", "checksums");
    for (int l = 0; l < 3; l++) {
        const qlayer_t *L = &LAYERS[l];
        long long c1 = 0, c2 = 0, c3 = 0, c4 = 0;
        double a = bench_v1(L, &c1);
        double b = bench_v2(L, &rt[l], &c2);
        double d = bench_v3(L, &rt[l], &c3);
        double e = bench_v4(L, &rt[l], &c4);
        printf("L%d (%2d->%2d,%4d MAC) %9.4f %9.4f %9.4f %9.4f     %5.2fx    [%s]\n",
               l, L->in_dim, L->out_dim, L->in_dim * L->out_dim,
               a, b, d, e, a / e,
               (c1 == c2 && c2 == c3 && c3 == c4) ? "all identical" : "MISMATCH!");
    }
    return 0;
}
