# -*- coding: utf-8 -*-
"""
read_vofa.py  --  AI 工具（非学习本体）
=========================================
直接读取 F28P55x 发出的 VOFA "JustFloat" 串口帧，统计关键通道。
用途：板端性能/精度实验时，AI 直接自己读串口分析，不再手抄 VOFA 数字。

帧格式：N 个 float32（小端）+ 4 字节帧尾 00 00 80 7F
本工程 N=33，通道布局见下方 LAYOUT33。

[WARN] 串口独占：运行前先关闭 VOFA+ / 其他串口工具。

用法（torch 环境 python）：
  E:\anaconda\envs\torch\python.exe read_vofa.py                      # COM6 默认 3 秒
  E:\anaconda\envs\torch\python.exe read_vofa.py --frames 300 --dump
  E:\anaconda\envs\torch\python.exe read_vofa.py --seconds 5 --csv board.csv
"""
import argparse
import struct
import sys
import time

TAIL = b"\x00\x00\x80\x7f"

LAYOUT = """已知通道布局 (36ch, C4 诊断版)：
  CH0..9    锯齿波输入          CH10..19  NPU 输出
  CH20      NPU 耗时 (us/10点)  CH21      CPU 耗时 (us/10点)
  CH22      自检最大误差         CH23..32  手写 CPU 输出 (10点)
  CH33..35  C4 诊断: 每层周期数 L0/L1/L2 (单点)"""


def parse_frames(buf, nch):
    """切出所有完整帧 -> (frames, frame_len)；用等差数列滤掉误检的帧尾"""
    fl = nch * 4 + 4
    tails = []
    i = buf.find(TAIL)
    while i != -1:
        tails.append(i)
        i = buf.find(TAIL, i + 1)
    if not tails:
        return [], fl
    best, run = [], [tails[0]]
    for t in tails[1:]:
        d = t - run[-1]
        if d == fl:
            run.append(t)
        elif d > fl:
            if len(run) > len(best):
                best = run
            run = [t]
        # d < fl：两帧尾太近 = 误检，跳过
    if len(run) > len(best):
        best = run
    frames = []
    for t in best:
        if t - nch * 4 < 0:
            continue  # 缓冲区开头不完整
        frames.append(struct.unpack_from("<%df" % nch, buf, t - nch * 4))
    return frames, fl


def col(frames, ch):
    return [f[ch] for f in frames]


def mean(v):
    return sum(v) / len(v)


def verdict22(v):
    if abs(v + 1.0) < 1e-6:
        return "[FAIL] 自检没执行（还是初值 -1）"
    if v < -0.5:
        return "[FAIL] 异常负值"
    if 0.005 <= v <= 0.08:
        return "[OK]   正常范围（期望 ~3.1e-2）"
    if 0.08 < v <= 2.0:
        return "[FAIL] 偏大：疑似魔数被 -O2 折叠（0.1~1.0 特征）"
    return "[WARN] 异常值，人工检查"


def main():
    ap = argparse.ArgumentParser(description="读取 VOFA JustFloat 帧并统计（AI 工具）")
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--channels", type=int, default=36)
    ap.add_argument("--seconds", type=float, default=3.0, help="最长读取时间（秒）")
    ap.add_argument("--frames", type=int, default=0, help="目标帧数（达到提前结束；0=纯按时间）")
    ap.add_argument("--dump", action="store_true", help="打印一帧完整各通道")
    ap.add_argument("--csv", default="", help="把所有帧存成 CSV")
    ap.add_argument("--mhz", type=float, default=150.0, help="SYSCLK MHz（算 cycles/MAC）")
    ap.add_argument("--macs", type=int, default=4224, help="每点 MAC 数（1->64->64->1）")
    a = ap.parse_args()

    import serial  # pyserial

    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.1)
    except Exception as e:
        print("[FAIL] 打不开 %s: %s" % (a.port, e))
        print("       -> 串口被独占？先关闭 VOFA+ 再运行")
        return 1

    print("读取 %s @ %d ...（%.1f 秒%s）" % (
        a.port, a.baud, a.seconds,
        "，或 %d 帧" % a.frames if a.frames else ""))
    buf = bytearray()
    fl = a.channels * 4 + 4
    t0 = time.time()
    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)
            if time.time() - t0 >= a.seconds:
                break
            if a.frames and len(buf) >= a.frames * fl:
                break
    finally:
        ser.close()
    dt = time.time() - t0

    frames, _ = parse_frames(bytes(buf), a.channels)
    if not frames:
        print("[FAIL] 收到 %d 字节，但没解析出完整帧" % len(buf))
        print("       -> 板子在发吗？波特率/通道数对吗？（当前 --channels %d）" % a.channels)
        return 1

    print("收到 %d 字节 -> %d 帧    用时 %.2f s    帧率 %.1f Hz"
          % (len(buf), len(frames), dt, len(frames) / dt))
    if a.channels >= 33:
        print(LAYOUT)

    if a.channels >= 23:
        v20, v21, v22 = col(frames, 20), col(frames, 21), col(frames, 22)
        m20, m21, m22 = mean(v20), mean(v21), mean(v22)
        print("")
        print("CH20 NPU 10点耗时 : %8.1f us   (min %.1f / max %.1f)" % (m20, min(v20), max(v20)))
        print("CH21 CPU 10点耗时 : %8.1f us   (min %.1f / max %.1f)" % (m21, min(v21), max(v21)))
        if m20 > 0:
            print("  -> 比值 CPU/NPU = %.2fx   每点: NPU %.1f us / CPU %.1f us"
                  % (m21 / m20, m20 / 10.0, m21 / 10.0))
        if a.macs > 0:
            # 注意：CH20/21 是"10 个点"的耗时 -> 总 MAC 数 = 10 * 每点MAC
            print("  -> 周期/MAC @%.0fMHz (每点 %d MAC): NPU %.1f / CPU %.1f"
                  % (a.mhz, a.macs,
                     m20 * a.mhz / (10.0 * a.macs), m21 * a.mhz / (10.0 * a.macs)))
        print("CH22 自检最大误差 : %.4f   %s" % (m22, verdict22(m22)))

    if a.channels >= 36:
        c33 = mean(col(frames, 33))
        c34 = mean(col(frames, 34))
        c35 = mean(col(frames, 35))
        print("")
        print("C4 每层周期数 (单点):")
        print("  L0 (1->64,    64 MAC) : %8.0f cyc -> %6.1f cyc/行" % (c33, c33 / 64.0))
        print("  L1 (64->64, 4096 MAC) : %8.0f cyc -> %6.2f cyc/MAC" % (c34, c34 / 4096.0))
        print("  L2 (64->1,    64 MAC) : %8.0f cyc" % (c35,))
        if a.channels >= 23:
            per_pt = mean(col(frames, 21)) * a.mhz / 10.0
            rest = per_pt - (c33 + c34 + c35)
            print("  三层合计 %.0f cyc/点 | CPU 总 %.0f cyc/点 | 其余(量化/调用等) %.0f"
                  % (c33 + c34 + c35, per_pt, rest))

    if a.dump:
        f0 = frames[-1]
        print("")
        print("最后一帧全通道：")
        for c in range(a.channels):
            print("  CH%02d = %12.6f" % (c, f0[c]), end="")
            if c % 3 == 2:
                print("")
        print("")

    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as fp:
            fp.write(",".join("ch%d" % c for c in range(a.channels)) + "\n")
            for f in frames:
                fp.write(",".join("%.6f" % v for v in f) + "\n")
        print("已保存 CSV: %s（%d 帧）" % (a.csv, len(frames)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
