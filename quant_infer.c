#include "quant_params.h"
#include "golden_test.h"
#include <stdio.h>
#include <math.h>

#include <time.h>


///* 运行期状态: 由常量参数在"加载阶段"预先算出, 推理时直接取用 */
typedef struct {
    double M[64];       
}   layer_rt_t;

/* 魔数舍入: 与 nearbyint 逐位等价, 但编译成 3 条指令而非 libm 调用.
   原理: 加 1.5*2^52 把尾数对齐到整数位, FPU 默认舍入模式(就近取偶)自动取整.
   有效范围 |x| < 2^51.  (PC/x86 技巧, 板端用纯整数舍入) */
static inline int32_t round_nearest_even(double x)
{
    const double MAGIC = 6755399441055744.0;   /* 1.5 * 2^52 */
    return (int32_t)((x + MAGIC) - MAGIC);
}

static void prepare(const qlayer_t *L, layer_rt_t *rt)
{
    for (int o = 0; o < L->out_dim; o++)
    {
        rt->M[o] = L->S_in * (double)L->S_w[o] / L->S_out;
    }
}

static void quant_layer(const qlayer_t *L, const layer_rt_t *rt, const int8_t *x, int8_t *q_out)
{
    for (int32_t o = 0; o < L->out_dim; o++)
    {
        //原先有问题的地方int32_t acc = L->b_int[o];
        int32_t acc = L->b_int[o];
        for (int32_t i = 0; i <L->in_dim; i++)
        {
            //原先由问题的地方 acc += ((int32_t)x[i] - L->z_in) * (int32_t)L->W_q[o * L->in_dim + i];
            acc += ((int32_t)x[i] - L->z_in) * (int32_t)L->W_q[o * L->in_dim + i];
        }
        double M = rt->M[o];
        int32_t r = round_nearest_even(acc * M) + L->z_out;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        q_out[o] = (int8_t)r;
    }
}



int main(void)
{
    /* ---- 块 1: 工作区 ---- */
    
    int8_t q0[1], q1[64], q2[64], q3[1];
    layer_rt_t rt0, rt1, rt2;
    long   diff_q1 = 0, diff_q2 = 0;      /* 逐位对拍的最大差 */
    double diff_y  = 0.0;

    prepare(&LAYERS[0], &rt0);
    prepare(&LAYERS[1], &rt1);
    prepare(&LAYERS[2], &rt2);

    /* ---- 块 2: 逐点测试 (200 点) ---- */
    for (int n = 0; n < N_TEST; n++) {
        /* 2.1 量化输入 */
        q0[0] = (int8_t)((int32_t)nearbyint((double)G_S[n] / LAYERS[0].S_in)
                         + LAYERS[0].z_in);

        /* 2.2 三层前向 */
        quant_layer(&LAYERS[0], &rt0, q0, q1);
        quant_layer(&LAYERS[1], &rt1, q1, q2);
        quant_layer(&LAYERS[2], &rt2, q2, q3);
        
        double y = LAYERS[2].S_out * ((double)q3[0] - LAYERS[2].z_out);

        /* 第 1 层对拍: 每个通道比一次, 只留住最大的那个差 */
        for (int i = 0; i < 64; i++) {
            int d = (int)q1[i] - (int)G_Q1[n][i];     /* 先转 int 再减, 防 int8 溢出 */
            if (d < 0) d = -d;
            if (d > diff_q1) diff_q1 = d;
        }

        /* 第 2 层同理 */
        for (int i = 0; i < 64; i++) {
            int d = (int)q2[i] - (int)G_Q2[n][i];
            if (d < 0) d = -d;
            if (d > diff_q2) diff_q2 = d;
        }

        /* 输出层 (浮点, 用 fabs) */
        double dy = fabs(y - (double)G_Y[n]);
        if (dy > diff_y) diff_y = dy;
    }


    /* 打印三段结果 */
    printf("q1 max diff = %ld\n", diff_q1);
    printf("q2 max diff = %ld\n", diff_q2);
    printf("y  max diff = %.3e\n", diff_y);
    return 0;
}