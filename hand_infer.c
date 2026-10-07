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

#if defined(__TI_COMPILER_VERSION__)
#include "driverlib.h"          /* C4 诊断: 读 CPUTimer1 给每层计时 */
#endif

/* 供 CCS Expressions 窗口观察; volatile 保证不被优化掉 */
volatile float g_st_maxerr   = -1.0f;   /* 自检最大误差 (期望 ~3.1e-2) */
volatile float g_st_maxerr_s =  0.0f;   /* 最大误差出现的输入 s */
volatile float g_y_cpu       =  0.0f;   /* 最近一次 hand_infer_one 的输出 */

/* C4 诊断: 每层最近一次耗时 (周期数, 单点); 仅 TI 编译器下真实测量, PC 上恒 0 */
volatile uint32_t g_cyc_l0 = 0u, g_cyc_l1 = 0u, g_cyc_l2 = 0u;

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
    /* C1 实验: 循环下标用 16 位 (C28x 的 int/int16_t 就是 16 位).
     * 汇编依据 (docs/08): int32_t 下标每轮要 4 条循环控制 (MOVB/SUBB/CMPL/B);
     * 下标最大 = 63*64+63 = 4095, 16 位绰绰有余.
     * 累加器 acc 仍是 int32_t, 勿动! */
    int16_t o, i;
    /* C2 实验: 16x16 乘法 + 零点半减提出内层 (循环不变式).
     * 范围证明 (int8 语义: x,W_q ∈ [-128,127]; z_in ∈ {0,-128}):
     *   xs = x[i]-z_in ∈ [-128, 255]  (int16 容得下)
     *   乘积 ∈ [-32640, 32385] ⊂ int16 -> 16 位乘法无损 */
    static int16_t xs_buf[64];                  /* in_dim <= 64 */
    /* C5 实验: 两遍结构 —— 先把整层的整数点积全部算完, 再批量 requant.
     * 动机 (C4 诊断): 每行 "仪式" ~120 cyc, 其中 requant 的浮点依赖链
     * (MOV32->I32TOF32->MPYF32->ADDF32->ADDF32->F32TOI32, 每步 4 周期延迟)
     * 把 6 个 NOP 硬塞进每行; 拆成独立 item 的第二遍, 调度器可重叠延迟 */
    static int32_t acc_buf[64];
    for (i = 0; i < L->in_dim; i++)
        xs_buf[i] = (int16_t)(x[i] - (int16_t)L->z_in);

    /* 第 1 遍: 纯整数点积 (期待仍被编成 RPT + MAC) */
    for (o = 0; o < L->out_dim; o++)
    {
        const int16_t *w = &L->W_q[o * L->in_dim];
        int32_t acc = L->b_int[o];
        for (i = 0; i < L->in_dim; i++)
        {
            acc += (int32_t)(xs_buf[i] * w[i]);
        }
        acc_buf[o] = acc;
    }

    /* 第 2 遍: 批量 requant (独立 item; C5a: 单加魔数, 整数域再减去它) */
    {
        const float   *Mp = L->M;
        const int32_t *ap = acc_buf;
        int16_t       *qp = q_out;
        const int32_t  zm = L->z_out - 12582912;  /* z_out - MAGIC, 一次算好 */
        for (o = 0; o < L->out_dim; o++)
        {
            /* (int32_t)(y + 1.5*2^23) 截断 = 12582912 + round_even(y), 范围 |y|<2^22 */
            int32_t r = (int32_t)((float)(*ap++) * (*Mp++) + 12582912.0f) + zm;
            if (r < -128) r = -128;
            if (r >  127) r =  127;
            *qp++ = (int16_t)r;         /* int8 值存 16 位容器 (C28x 无 8 位类型) */
        }
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

#if defined(__TI_COMPILER_VERSION__)
    {
        uint32_t t0, t1, t2, t3;
        t0 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        hand_quant_layer(&LAYERS_MCU[0], q0, q1);
        t1 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        hand_quant_layer(&LAYERS_MCU[1], q1, q2);
        t2 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        hand_quant_layer(&LAYERS_MCU[2], q2, q3);
        t3 = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        g_cyc_l0 = t0 - t1;      /* 递减计数: 差值 = 经过周期数 (回绕安全) */
        g_cyc_l1 = t1 - t2;
        g_cyc_l2 = t2 - t3;
    }
#else
    hand_quant_layer(&LAYERS_MCU[0], q0, q1);
    hand_quant_layer(&LAYERS_MCU[1], q1, q2);
    hand_quant_layer(&LAYERS_MCU[2], q2, q3);
#endif

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


