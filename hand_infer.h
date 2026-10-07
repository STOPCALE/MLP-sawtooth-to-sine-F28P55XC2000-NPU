/* ============================================================================
 *  hand_infer.h -- 手写 int8 整数推理 (TI F28P55x) 对外接口
 *
 *  用法(在 empty_driverlib_main.c 里):
 *      #include "hand_infer.h"
 *      ...
 *      hand_selftest();              // 开机自检一次, 结果在 g_st_maxerr
 *      y = hand_infer_one(s);        // 每个采样点调一次
 * ==========================================================================*/
#ifndef HAND_INFER_H
#define HAND_INFER_H

#include <stdint.h>

/* 单个采样点: 浮点 s (归一化到 [-1,1]) -> 浮点 y */
float hand_infer_one(float s);

/* 自检: 128 点均匀扫 [-1,1], 与解析真值 -sin(pi*s) 比较
   结果写入下面的全局变量 (供 CCS Expressions 窗口观察) */
void hand_selftest(void);

extern volatile float g_st_maxerr;      /* 自检最大误差 (期望 ~3.1e-2) */
extern volatile float g_st_maxerr_s;    /* 最大误差出现的输入 s */
extern volatile float g_y_cpu;          /* 最近一次 hand_infer_one 的输出 (可选) */

#endif /* HAND_INFER_H */

/* === end of hand_infer.h === */
