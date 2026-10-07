/* ============================================================================
 *  empty_driverlib_main.c —— saw2sin 模型 F28P55 NPU 板端验证程序
 *
 *  功能（对应板端验证三条要求）：
 *   1. 单片机内部自行产生锯齿波输入数据（不依赖外部信号源）
 *   2. 每 10 个采样点为 1 个窗口, 送入 NPU 推理一次 (锯齿->正弦 逐点映射)
 *   3. 用 CPUTimer1 (150MHz 递减计数) 记录每一次推理耗时, 微秒数经串口发送
 *   4. 串口帧按顺序: 10 个输入 -> 10 个输出 -> 1 个推理耗时, 用 VOFA+ JustFloat
 *      协议 (float 小端 + 帧尾 00 00 80 7F), 可直接在 VOFA+ 看波形:
 *        CH0..CH9  = 模型输入 (锯齿波)
 *        CH10..CH19 = 模型输出 (正弦波)
 *        CH20      = 本次推理耗时 (us)
 *
 *  说明: 原 SCIA 回显测试程序备份为 empty_driverlib_main.c.bak
 *        NPU 模型库: artifacts/mod.a + artifacts/tvmgen_default.h (由 TI QAT + tvmc 编译生成)
 * ==========================================================================*/
#include "board.h"
#include "device.h"
#include "driverlib.h"
#include <stdint.h>
#include "hand_infer.h"

#include "artifacts/tvmgen_default.h"      /* NPU 模型接口 (mod.a 提供) */

// ----------------- 模型与信号参数 -----------------
#define FRAME            10                 /* 模型窗口 (1,1,10,1)          */
#define SYSCLK_MHZ       150.0f             /* 系统主频 (见 clocktree.h)    */
#define PHASE_STEP       (1.0f / 2000.0f)   /* 锯齿波 2000 个采样点/周期     */

// ----------------- 全局变量 (可在 CCS Expressions 窗口 Watch) -----------------
static float g_phase    = 0.0f;             /* 锯齿波相位 0..1              */
static float g_in [FRAME];                  /* 推理输入缓冲 (10 点)         */
static float g_out[FRAME];                  /* 推理输出缓冲 (10 点)         */
volatile uint32_t g_frame_count = 0;        /* 已推理帧数                   */
volatile uint32_t g_cycles_last = 0;        /* 最近一次推理周期数            */
volatile uint32_t g_cycles_min  = 0xFFFFFFFFu;  /* 最小周期数 (最快一次)    */
volatile uint32_t g_cycles_max  = 0;        /* 最大周期数 (最慢一次)        */
volatile float    g_us_last     = 0.0f;     /* 最近一次推理耗时 (微秒)       */

volatile uint32_t g_cycles_cpu = 0;        /* 手写 CPU 推理(10 点)的周期数 */
volatile float    g_us_cpu     = 0.0f;     /* 手写 CPU 推理(10 点)的微秒数 */
static   float    g_ycpu[FRAME];           /* 手写 CPU 推理的 10 点输出 (与 NPU 的 g_out 同口径) */

static struct tvmgen_default_inputs  g_npu_in;    /* 模型输入结构 (含指针) */
static struct tvmgen_default_outputs g_npu_out;   /* 模型输出结构 (含指针) */

// ----------------- 函数声明 -----------------
static void CPUTimer1_init(void);
static void SCI_sendFloat(float v);
static void SCI_sendFrameTail(void);

void main(void)
{
    uint32_t t_start, t_end, cycles;
    uint16_t k;

    // ---- 1. 系统初始化 (与工程原有流程一致) ----
    Device_init();                 // 时钟树 + 所有外设时钟 (device.c 中已含 NPU 时钟使能)
    Interrupt_initModule();
    Interrupt_initVectorTable();
    Board_init();                  // SysConfig 生成 (含 SCI_init: SCIA 115200-8N1-FIFO)
    EINT;
    ERTM;

    // ---- 2. 计时器初始化: CPUTimer1, SYSCLK 150MHz, 自由递减计数 ----
    //   (先于自检: C4 诊断在 hand_infer_one 里读它给每层计时)
    CPUTimer1_init();

    // ---- 3.5 手写整数推理自检 (开机一次) ----
    //   结果看 g_st_maxerr: 期望 ~3.1e-2  (与 PC 端 PTQ 实测一致)
    //   若变成 1e-1 ~ 1.0 => 魔数取整被编译器优化掉了
    hand_selftest();

    // ---- 3. 绑定模型输入输出指针 ----
    g_npu_in.input   = (void *)g_in;
    g_npu_out.output = (void *)g_out;

    // ---- 4. 留 0.5s 给 XDS110 虚拟串口枚举, 避免首批帧丢失 ----
    DEVICE_DELAY_US(500000);

    // ---- 5. 主循环: 生成数据 -> NPU 推理 -> 计时 -> 发送 VOFA 帧 ----
    while (1)
    {
        // (a) 内部生成 10 个锯齿采样: s = 2*phase - 1, phase 在 0..1 循环推进
        for (k = 0u; k < FRAME; k++)
        {
            g_in[k] = 2.0f * g_phase - 1.0f;
            g_phase += PHASE_STEP;
            if (g_phase >= 1.0f)
            {
                g_phase -= 1.0f;
            }
        }

        // (b) NPU 推理 + 计时: 计测窗口 = run() 调用 + 等待完成标志
        t_start = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        (void)tvmgen_default_run(&g_npu_in, &g_npu_out);
        while (!tvmgen_default_finished)
        {
            /* 轮询等待 NPU 子图完成 (完成标志由 NPU 中断服务程序置位) */
        }
        t_end = CPUTimer_getTimerCount(CPUTIMER1_BASE);
    
        cycles = t_start - t_end;      // 递减计数: 差值 = 经过周期数 (无符号回绕安全)
        g_cycles_last = cycles;
        if (cycles < g_cycles_min) { g_cycles_min = cycles; }
        if (cycles > g_cycles_max) { g_cycles_max = cycles; }
        g_us_last = (float)cycles / SYSCLK_MHZ;    // 周期数 -> 微秒
        g_frame_count++;

        // (b2) 手写 CPU 推理 + 计时 -- 10 个点, 与 NPU 同口径可直接比较
        t_start = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        for (k = 0u; k < FRAME; k++)
        {
            g_ycpu[k] = hand_infer_one(g_in[k]);
        }
        t_end = CPUTimer_getTimerCount(CPUTIMER1_BASE);
        g_cycles_cpu = t_start - t_end;
        g_us_cpu = (float)g_cycles_cpu / SYSCLK_MHZ;
        g_y_cpu  = g_ycpu[FRAME - 1];       /* 供 CCS Expressions 观察 */

        // (c) 发送一帧 (VOFA+ JustFloat)
        //     铁律: 所有数据必须在帧尾之前, 否则下一帧整体错位一格!
        //     CH0 ..CH9   锯齿波输入
        //     CH10..CH19  NPU 输出 (正弦)
        //     CH20        NPU 一次推理(10 个点)耗时 [us]
        //     CH21        手写 CPU 一次推理(10 个点)耗时 [us]   <-- 与 CH20 同口径
        //     CH22        自检最大误差 (期望 ~3.1e-2; 若 0.1~1.0 => 魔数被优化折叠)
        //     CH23..CH32  手写 CPU 的 10 点输出                    <-- 与 CH10..19 逐点对比
        //     CH33..CH35  C4 诊断: 每层耗时 [周期数] (L0 / L1 / L2, 单点)
        for (k = 0u; k < FRAME; k++) { SCI_sendFloat(g_in[k]);  }
        for (k = 0u; k < FRAME; k++) { SCI_sendFloat(g_out[k]); }
        SCI_sendFloat(g_us_last);
        SCI_sendFloat(g_us_cpu);
        SCI_sendFloat(g_st_maxerr);
        for (k = 0u; k < FRAME; k++) { SCI_sendFloat(g_ycpu[k]); }
        SCI_sendFloat((float)g_cyc_l0);
        SCI_sendFloat((float)g_cyc_l1);
        SCI_sendFloat((float)g_cyc_l2);
        SCI_sendFrameTail();
    }
}

// CPUTimer1 初始化: 最大周期、/1 预分频、开始递减
//   读取方式: t0 = 读取; ...被测代码...; t1 = 读取; 耗时周期数 = t0 - t1
static void CPUTimer1_init(void)
{
    CPUTimer_stopTimer(CPUTIMER1_BASE);
    CPUTimer_setPeriod(CPUTIMER1_BASE, 0xFFFFFFFFu);   /* 最大周期 (约 28.6s 回绕) */
    CPUTimer_setPreScaler(CPUTIMER1_BASE, 0u);         /* TDDR=0 -> /1 (150MHz)    */
    CPUTimer_reloadTimerCounter(CPUTIMER1_BASE);       /* 计数器 = period          */
    CPUTimer_startTimer(CPUTIMER1_BASE);
}

// 发送一个 float: 拆成 4 字节小端 (VOFA JustFloat 协议的基本单元)
static void SCI_sendFloat(float v)
{
    union { float f; uint32_t u; } conv;
    conv.f = v;
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)( conv.u        & 0xFFu));
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)((conv.u >> 8)  & 0xFFu));
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)((conv.u >> 16) & 0xFFu));
    SCI_writeCharBlockingFIFO(mySCI0_BASE, (uint16_t)((conv.u >> 24) & 0xFFu));
}

// VOFA+ JustFloat 帧尾: 0x00 0x00 0x80 0x7F
static void SCI_sendFrameTail(void)
{
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x00u);
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x00u);
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x80u);
    SCI_writeCharBlockingFIFO(mySCI0_BASE, 0x7Fu);
}

// 串口已回显的字符个数 (原回显程序遗留变量, 已不再使用, 保留防止外部引用报错)
uint32_t echoCount = 0;

// 原回显程序的字符串发送函数 (已不再使用; 保留作参考, 备份见 .bak 文件)
void SCI_sendString(const char *msg)
{
    uint16_t len = 0;
    while (msg[len] != '\0')
    {
        len++;
    }
    SCI_writeCharArray(mySCI0_BASE, (uint16_t *)msg, len);
}

