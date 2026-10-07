#include "quant_params.h"
#include "golden_test.h"
#include <math.h>

static void quant_layer(const qlayer_t *L,const int8_t *x, int8_t *q_out)
{
    for (int32_t o = 0; o < L->out_dim; o++)
    {
        int32_t acc = L->b_int[o];
        for (int32_t i = 0; i <L->in_dim; i++)
        {
            acc += ((int32_t)x[i] - L->z_in) * (int32_t)L->W_q[o * L->in_dim + i];
        }
        double M = L->S_in * (double)L->S_w[o] / L->S_out;
        int32_t r = (int32_t)nearbyint(acc * M) + L->z_out;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        q_out[o] = (int8_t)r;
    }
}



int main(void)
{
    /* ---- 块 1: 工作区 ---- */
    
    int8_t q0[1], q1[64], q2[64], q3[1];
    long   diff_q1 = 0, diff_q2 = 0;      /* 逐位对拍的最大差 */
    double diff_y  = 0.0;

    /* ---- 块 2: 逐点测试 (200 点) ---- */
    for (int n = 0; n < N_TEST; n++) {
        /* 2.1 量化输入 */
        q0[0] = (int8_t)((int32_t)nearbyint((double)G_S[n] / LAYERS[0].S_in)
                         + LAYERS[0].z_in);

        /* 2.2 三层前向 */
        quant_layer(&LAYERS[0], q0, q1);
        quant_layer(&LAYERS[1], q1, q2);
        quant_layer(&LAYERS[2], q2, q3);
        
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