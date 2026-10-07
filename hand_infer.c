/* ============================================================================
 *  hand_infer.c -- 手写 int8 整数推理 (TI F28P55x, 纯 CPU/FPU)
 *
 *  来源: Charge_Num 工程的手写推理 (quant_infer.c / quant_infer.py), 移植要点:
 *    1) M 已在编译期算好 (quant_params_mcu.h 的 Lx_M[]), 运行时零初始化、零除法
 *    2) 全部用 float32 -- C2000 的 FPU 是单精度; double 是软件模拟(慢 10~50x)
 *    3) 魔数取整的魔数改为 1.5*2^23 (float 尾数 23 位)
 *    4) 累加器必须 int32_t (C2000 的 int 是 16 位!)
 *
 *  对外接口见 hand_infer.h
 * ==========================================================================*/
#include "hand_infer.h"
#include "quant_params_mcu.h"
#include <math.h>

/* 供 CCS Expressions 窗口观察; volatile 保证不被优化掉 */
volatile float g_st_maxerr   = -1.0f;   /* 自检最大误差 (期望 ~3.1e-2) */
volatile float g_st_maxerr_s =  0.0f;   /* 最大误差出现的输入 s */
volatile float g_y_cpu       =  0.0f;   /* 最近一次 hand_infer_one 的输出 */

#define N_SELFTEST 128
#define PI_F       3.14159265f

/* --------------------------------------------------------------------------
 * 魔数取整: 与"就近取偶"等价, 有效范围 |x| < 2^23
 *   加 1.5*2^23 把尾数对齐到整数位, FPU 默认舍入模式(就近取偶)自动取整,
 *   再减回去即得整数. 全程 float, 硬件 FPU 三条指令, 不调用数学库.
 * ------------------------------------------------------------------------ */
static inline int32_t round_nearest_even_f(float x)
{
    const float MAGIC = 12582912.0f;           /* 1.5 * 2^23 */
    return (int32_t)((x + MAGIC) - MAGIC);
}

/* --------------------------------------------------------------------------
 * 单层整数前向 (与 PC 版逻辑逐一对应)
 *   acc = b_int[o] + sum_i (x[i] - z_in) * W_q[o][i]      <- int32 累加
 *   q   = clip( round(acc * M[o]) + z_out, -128, 127 )
 * ------------------------------------------------------------------------ */
static void hand_quant_layer(const qlayer_mcu_t *L, const int16_t *x, int16_t *q_out)
{
    int32_t o, i;
    for (o = 0; o < L->out_dim; o++)
    {
        int32_t acc = L->b_int[o];
        for (i = 0; i < L->in_dim; i++)
        {
            acc += ((int32_t)x[i] - L->z_in) * (int32_t)L->W_q[o * L->in_dim + i];
        }
        int32_t r = round_nearest_even_f((float)acc * L->M[o]) + L->z_out;
        if (r < -128) r = -128;
        if (r >  127) r =  127;
        q_out[o] = (int16_t)r;          /* int8 值存 16 位容器 (C28x 无 8 位类型) */
    }
}

/* --------------------------------------------------------------------------
 * 单个采样点: 浮点 s -> 浮点 y
 *   中间张量用 static buffer: 纯链条结构, 一层写、下一层读, 天然可复用
 * ------------------------------------------------------------------------ */
float hand_infer_one(float s)
{
    static int16_t q0[1], q1[64], q2[64], q3[1];

    /* 入口量化: round(s / S_in) + z_in, 先 clip 再装箱
       (s = +1.0 时 s/S_in ~ 127.5 -> 就近取偶会得 128, 不 clip 会绕圈) */
    int32_t t = round_nearest_even_f(s / LAYERS_MCU[0].S_in) + LAYERS_MCU[0].z_in;
    if (t < -128) t = -128;
    if (t >  127) t =  127;
    q0[0] = (int16_t)t;

    hand_quant_layer(&LAYERS_MCU[0], q0, q1);
    hand_quant_layer(&LAYERS_MCU[1], q1, q2);
    hand_quant_layer(&LAYERS_MCU[2], q2, q3);

    /* 出口反量化: y = S_out * (q - z_out) */
    return LAYERS_MCU[2].S_out * ((float)q3[0] - LAYERS_MCU[2].z_out);
}

/* --------------------------------------------------------------------------
 * 自检: 均匀扫 [-1, 1], 手写推理 vs 解析真值 y = -sin(pi*s)
 *
 * 判据: PC 上实测 PTQ int8 的 max 误差 = 3.14e-2
 *       -> 板端结果应落在同一量级; 若变成 1e-1 ~ 1.0, 说明取整/魔数出问题了
 * ------------------------------------------------------------------------ */
void hand_selftest(void)
{
    float maxerr = 0.0f, maxerr_s = 0.0f;
    int32_t n;

    for (n = 0; n < N_SELFTEST; n++)
    {
        float s  = -1.0f + 2.0f * (float)n / (float)(N_SELFTEST - 1);
        float y  = hand_infer_one(s);
        float yt = -sinf(PI_F * s);
        float e  = fabsf(y - yt);
        if (e > maxerr)
        {
            maxerr   = e;
            maxerr_s = s;
        }
    }

    g_st_maxerr   = maxerr;
    g_st_maxerr_s = maxerr_s;
}


