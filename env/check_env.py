#!/usr/bin/env python3
"""
mllm-lab 环境验证 + 基线测量

配套：D:\\AI\\W1_第一天执行清单.md 的 ⑥ / ⑥+ / ⑦ 三步合并实现。

用法：
    python env/check_env.py              # 验证 + 基准测试 + 写 baseline.json
    python env/check_env.py --no-bench   # 只做环境验证（不占显存，随时可跑）

退出码：0 = 全部通过；1 = 有检查项失败（方便接 CI）

为什么要有这个脚本：
    环境验证必须"可复现"。手敲一遍 python -c "..." 看过就丢，
    下次 OOM 或换机器时你没法回答"当时到底是什么状态"。

==========================================================================
修订历史 —— 两次都是"参考值取错"导致误判，值得记住
==========================================================================
v1  用绝对阈值 `bf16 < 50 TFLOPS => FAIL`。
    RTX 5060 Ti 稠密 bf16 峰值只有 47.4 TFLOPS，阈值比峰值还高 -> 满血卡被误判 FAIL。

v2  改用比值判据（bf16/fp32 >= 2.0，与型号无关），方向对。
    但仍用 `nvidia-smi --query-gpu=clocks.max.sm` 算"理论峰值" —— 又取错了：
    它返回的是**芯片能跑到的最高频率（3090 MHz）**，不是**这张卡的额定 boost（2572 MHz）**。
    按 3090 MHz 算出的 57.0 TFLOPS 谁都达不到 -> 满血卡被算成"83.9% of peak"。

v3  分母改成**实测的持续负载时钟**（边跑 matmul 边采 clocks.sm，取中位数）。
    这才是这张卡在 180W 功耗墙下真正能维持的频率，用它算出的上限才可比。
    同时采 power.draw，直接看出是不是撞了功耗墙。

教训（通用）：**任何"达成率"的分母都必须是这台机器真的能达到的值。**
    芯片 datasheet 上限、另一档硬件的经验区间，都不能当分母。

--------------------------------------------------------------------------
v4  修两个 bug —— 都由一次实跑暴露，都是我自己写错，不是机器问题：

    (1) 崩在 `TypeError: 'Event' object is not callable`
        LoadSampler 把成员起名 `self._stop = threading.Event()`，
        **遮蔽了 threading.Thread._stop() 方法**。Thread.join() 内部会回调
        self._stop() 来释放 tstate lock，撞上 Event 对象就炸。
        ->> 改名 `self._stop_evt`。
        教训：**别用基类已有的（尤其下划线开头的）名字当成员名。**

    (2) fp16 测出 `-17.15 ms / -8.0 TFLOPS`（负的时间！）
        计时用了 `time.time()` —— 它**不是单调时钟**。WSL2 是虚拟机，
        宿主会做 NTP 校正/时钟步进，`time.time()` 可能**回跳**，
        于是 t1 < t0，dt 为负。
        ->> 改用 **CUDA Event**（GPU 自己的计时器，单调）做主机侧循环计时，
            并且每个基准**测 3 次取中位数**（中位数天然免疫单次抖动）。
            带宽/持续负载这类主机侧计时改用 `time.perf_counter()`（单调）。
        教训：**测性能只用单调时钟；能落到设备事件的就别用主机墙钟。**
==========================================================================
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "baseline.json")

EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_PY = "3.11"
REQUIRED_ARCH = "sm_120"       # Blackwell。缺了它第一个算子就会爆 no kernel image
DESKTOP_MEM_WARN_MIB = 900     # WDDM 下桌面占超过这个值就提醒清理

TENSOR_RATIO = 2.0             # 稠密 bf16/fp16 Tensor 吞吐 = 2 x FP32 向量吞吐
FP32_CORES_PER_SM = 128
RATIO_MIN = 2.0                # 低于它 = 没走 Tensor Core
PEAK_PCT_WARN = 85.0           # 实测/上限 < 85% = kernel 没打满（不是降频）
LOAD_SECONDS = 4.0             # 持续负载采样时长
SAMPLE_INTERVAL = 0.25         # 时钟/功耗采样间隔（秒）

MEMBW_SPEC_GBPS = 448.0        # 参考值：RTX 5060 Ti 128-bit GDDR7 28Gbps


def sec(title):
    print("\n" + "=" * 62)
    print("  " + title)
    print("=" * 62)


def mib(nbytes):
    return round(nbytes / 1024 ** 2, 1)


def gib(nbytes):
    return round(nbytes / 1024 ** 3, 3)


def _smi(query, nounits=False):
    try:
        fmt = "--format=csv,noheader" + (",nounits" if nounits else "")
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, fmt],
            capture_output=True, text=True, timeout=15,
        )
        return (out.stdout or out.stderr).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def _f(x, default=None):
    try:
        return float(str(x).strip())
    except Exception:
        return default


def sm_clock_max_mhz():
    """芯片能跑到的最高 SM 频率 —— 仅供参照，**不能**用来算达成率。"""
    return _f(_smi("clocks.max.sm", nounits=True).split()[0])


class LoadSampler(threading.Thread):
    """持续负载期间后台采 SM 时钟 / 功耗。

    为什么要另起线程：主线程在跑 matmul，Python 侧同步调用 nvidia-smi 会打断它。
    """

    QUERY = "clocks.sm,power.draw,power.limit"

    def __init__(self, interval=SAMPLE_INTERVAL):
        super().__init__(daemon=True)
        self.interval = interval
        self.clocks = []
        self.powers = []
        self.limits = []
        # ⚠️ 别叫 self._stop —— 那会遮蔽 threading.Thread._stop()，
        #    Thread.join() 内部回调它时会抛 TypeError: 'Event' object is not callable
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            raw = _smi(self.QUERY, nounits=True)
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) >= 3:
                c, p, l = _f(parts[0]), _f(parts[1]), _f(parts[2])
                if c:
                    self.clocks.append(c)
                if p is not None:
                    self.powers.append(p)
                if l is not None:
                    self.limits.append(l)
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=3)

    def median_clock(self):
        return statistics.median(self.clocks) if self.clocks else None

    def mean_power(self):
        return statistics.mean(self.powers) if self.powers else None

    def limit(self):
        return statistics.median(self.limits) if self.limits else None


def peak_tflops(sm, clk_mhz, fp32_only=False):
    """按给定时钟算理论上限（TFLOPS）。clk_mhz 用**实测**值才有意义。"""
    if not sm or not clk_mhz:
        return None
    ratio = 1.0 if fp32_only else TENSOR_RATIO
    return sm * FP32_CORES_PER_SM * 2 * ratio * clk_mhz * 1e6 / 1e12


# ----------------------------------------------------------------------------
# 1. 环境验证
# ----------------------------------------------------------------------------
def check_env():
    sec("1. 环境验证")
    r = {}
    fails = []

    r["python"] = sys.version.split()[0]
    r["python_exe"] = sys.executable
    r["torch"] = torch.__version__
    r["torch_path"] = os.path.dirname(torch.__file__)
    r["cuda"] = getattr(torch.version, "cuda", None)
    r["cudnn"] = torch.backends.cudnn.version()
    r["arch_list"] = torch.cuda.get_arch_list()
    available = torch.cuda.is_available()
    r["cuda_available"] = available

    print("  python     : %s" % r["python"])
    print("  python exe : %s" % r["python_exe"])
    print("  torch      : %s" % r["torch"])
    print("  torch path : %s" % r["torch_path"])
    print("  cudnn      : %s" % r["cudnn"])
    print("  cuda       : %s" % available)

    # --- 检查 1：必须跑在 mllm 环境里（不是 base / 不是系统 python）
    if "envs/mllm" in r["python_exe"].replace("\\", "/"):
        print("  [OK]   运行在 envs/mllm 环境内")
    else:
        msg = "python 不在 envs/mllm 里 —— 环境激活错了，torch 会装错地方"
        fails.append(msg)
        print("  [FAIL] %s" % msg)

    # --- 检查 2：Python 版本
    if r["python"].startswith(EXPECTED_PY + "."):
        print("  [OK]   Python %s.x" % EXPECTED_PY)
    else:
        msg = "Python 是 %s，期望 %s.x（ML 生态兼容性最宽的区间）" % (r["python"], EXPECTED_PY)
        fails.append(msg)
        print("  [FAIL] %s" % msg)

    # --- 检查 3：torch 版本必须是 cu130
    if r["torch"] == EXPECTED_TORCH:
        print("  [OK]   torch == %s" % EXPECTED_TORCH)
    else:
        msg = "torch 是 %s，期望 %s（cu126 无 sm_120；cu128 会被锁在 2.12）" % (
            r["torch"], EXPECTED_TORCH)
        fails.append(msg)
        print("  [FAIL] %s" % msg)

    if not available:
        msg = "torch.cuda.is_available() == False —— 装成 CPU 版了"
        fails.append(msg)
        print("  [FAIL] %s" % msg)
        return r, fails
    print("  [OK]   cuda.is_available() == True")

    # --- 检查 4（关键）：arch_list 里必须有 sm_120
    cap = torch.cuda.get_device_capability()
    r["capability"] = list(cap)
    print("  cap        : %s   (检的是显卡，永远返回 (12,0)，不作数)" % (cap,))
    print("  arch_list  : %s" % r["arch_list"])
    if REQUIRED_ARCH in r["arch_list"]:
        print("  [OK]   arch_list 含 %s -> wheel 真的编了 Blackwell kernel" % REQUIRED_ARCH)
    else:
        msg = "arch_list 里没有 %s：%s —— 跑第一个算子就会爆 no kernel image" % (
            REQUIRED_ARCH, r["arch_list"])
        fails.append(msg)
        print("  [FAIL] %s" % msg)

    # --- 设备信息
    props = torch.cuda.get_device_properties(0)
    r["device"] = props.name
    r["sm_count"] = getattr(props, "multi_processor_count", None)
    r["mem_total_mib"] = round(props.total_memory / 1024 ** 2, 1)
    print("  device     : %s" % props.name)
    print("  sm count   : %s" % r["sm_count"])
    print("  mem total  : %.0f MiB (%s GiB)" % (
        props.total_memory / 1024 ** 2, gib(props.total_memory)))

    clk_max = sm_clock_max_mhz()
    r["sm_clock_max_mhz"] = clk_max
    print("  sm clk max : %s MHz  (芯片上限，仅供参考；达成率的分母要用实测时钟)"
          % clk_max)

    # --- 桌面占用（这条决定你这次实验能开多大）
    free_b, total_b = torch.cuda.mem_get_info()
    r["mem_free_mib"] = mib(free_b)
    r["mem_desktop_mib"] = mib(total_b - free_b)
    print("  mem free   : %s MiB" % r["mem_free_mib"])
    print("  desktop 占 : %s MiB" % r["mem_desktop_mib"])
    if r["mem_desktop_mib"] > DESKTOP_MEM_WARN_MIB:
        print("  [WARN] 桌面占用 %s MiB > %s MiB" % (
            r["mem_desktop_mib"], DESKTOP_MEM_WARN_MIB))
        print("         先暂停 Wallpaper Engine、关浏览器和 Zotero 再跑实验")

    smi = _smi("driver_version,name,memory.used,memory.total")
    r["nvidia_smi_raw"] = smi
    parts = [p.strip() for p in smi.split(",")]
    r["driver"] = parts[0] if parts else None
    print("  nvidia-smi : %s" % smi)

    return r, fails


# ----------------------------------------------------------------------------
# 2. 基准测试
# ----------------------------------------------------------------------------
def _timed_matmul(a, b, warmup, iters):
    """用 CUDA Event 计时 —— **不用 time.time()**。

    为什么：`time.time()` 不是单调时钟。WSL2 是虚拟机，宿主会 NTP 校正/步进时钟，
    它可能**回跳**，于是 (t1 - t0) 变负 —— 实测出现过 -17.15 ms / -8.0 TFLOPS。
    CUDA Event 用的是 GPU 自己的单调计时器，不受宿主时钟影响。

    返回：单次 matmul 的秒数。
    """
    for _ in range(warmup):
        a @ b
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters       # elapsed_time 单位是毫秒


def bench_matmul(dtype, n=4096, warmup=5, iters=50, repeats=3):
    """返回 (秒/次, TFLOPS)。

    v4：测 `repeats` 次取**中位数** —— 中位数天然免疫单次抖动
    （时钟步进、其他进程抢 GPU、偶发调度延迟）。
    """
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    dts = []
    for _ in range(repeats):
        dt = _timed_matmul(a, b, warmup, iters)
        if dt > 0:                       # 负数/零直接丢弃（有 repeats 兜底）
            dts.append(dt)
    if not dts:
        return None, None, None

    dt = statistics.median(dts)
    spread = (max(dts) - min(dts)) / dt * 100.0
    del a, b
    torch.cuda.empty_cache()
    return dt, 2.0 * n ** 3 / dt / 1e12, spread


def bench_membw(nbytes=256 * 1024 * 1024, iters=20):
    """D2D 拷贝。

    ⚠️ 读法：一次拷贝要搬 2N 字节（读 N + 写 N）。
       - copy_gbps = N/t   「应用层拷贝速率」——只有带宽的一半，别拿去比规格值
       - effective = 2N/t  「实际带宽」——这才该和 448 GB/s 比（73% 左右正常）

    v4：改用 time.perf_counter()（单调），不用 time.time()。
    """
    x = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    del x, y
    torch.cuda.empty_cache()
    gib_per_s = nbytes / 1024 ** 3 / dt
    return dt, gib_per_s, 2 * gib_per_s


def bench_sustained_load(seconds=LOAD_SECONDS):
    """持续跑 bf16 matmul，同时采 SM 时钟 / 功耗。

    这是 v3 的关键：拿到的**持续时钟**才是这张卡在功耗墙下真能维持的频率，
    用它算出的理论上限才配当"达成率"的分母。
    """
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()

    sampler = LoadSampler()
    sampler.start()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
    sampler.stop()
    del a, b
    torch.cuda.empty_cache()

    clk = sampler.median_clock()
    print("  持续负载 %ss 采样 %d 次：SM 时钟中位数 %s MHz / 功耗均值 %s W (限制 %s W)"
          % (seconds, len(sampler.clocks),
             "%.0f" % clk if clk else "n/a",
             "%.1f" % sampler.mean_power() if sampler.mean_power() else "n/a",
             "%.0f" % sampler.limit() if sampler.limit() else "n/a"))
    return clk, sampler.mean_power(), sampler.limit()


def bench():
    sec("2. 基准测试（这些数字是所有后续实验的参考线）")
    r = {}
    torch.cuda.reset_peak_memory_stats()

    for name, dtype, warm, it in (
        ("bf16", torch.bfloat16, 5, 50),
        ("fp16", torch.float16, 5, 50),
        ("fp32", torch.float32, 3, 20),
    ):
        print("  %s matmul 4096^3 ..." % name)
        dt, tf, spread = bench_matmul(dtype, warmup=warm, iters=it)
        if tf is None:
            print("    [FAIL] 计时结果全为负/零，无法测出有效值")
            r["%s_ms" % name] = None
            r["%s_tflops" % name] = None
            continue
        r["%s_ms" % name] = round(dt * 1000, 2)
        r["%s_tflops" % name] = round(tf, 1)
        r["%s_spread_pct" % name] = round(spread, 1)
        print("    %.2f ms   %.1f TFLOPS   (±%.1f%% 3 次离散度)" % (dt * 1000, tf, spread))

    # --- bf16 复测：与首轮对比，用来区分「天生慢」和「跑热了降频」
    print("  bf16 matmul 复测 (检测降频) ...")
    dt, tf, _ = bench_matmul(torch.bfloat16, warmup=5, iters=50)
    if tf is None:
        print("    [FAIL] 复测无效")
        r["bf16_repeat_tflops"] = None
    elif r.get("bf16_tflops"):
        r["bf16_repeat_tflops"] = round(tf, 1)
        drift = (tf - r["bf16_tflops"]) / r["bf16_tflops"] * 100.0
        r["bf16_drift_pct"] = round(drift, 1)
        print("    %.2f ms   %.1f TFLOPS   (与首轮差 %+.1f%%)" % (dt * 1000, tf, drift))

    print("  持续负载采样（用于确定达成率的分母）...")
    clk, power, limit = bench_sustained_load()
    r["sm_clock_load_mhz"] = round(clk, 1) if clk else None
    r["power_load_w"] = round(power, 1) if power else None
    r["power_limit_w"] = round(limit, 1) if limit else None

    print("  device-to-device 拷贝 (256 MB) ...")
    dt, copy_gbps, eff_gbps = bench_membw()
    r["membw_copy_gbps"] = round(copy_gbps, 1)
    r["membw_effective_gbps"] = round(eff_gbps, 1)
    print("    应用层拷贝 %.0f GB/s ／ 实际带宽 %.0f GB/s (%.0f%% of %.0f GB/s 规格)"
          % (copy_gbps, eff_gbps, eff_gbps / MEMBW_SPEC_GBPS * 100, MEMBW_SPEC_GBPS))

    r["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    print("  本测试峰值显存 : %s MiB" % r["peak_mem_mib"])
    return r


def verdict(env, bench_r):
    """判据全部基于比值 / 百分比，且分母用**实测**时钟 —— 不写死型号绝对数值。"""
    sec("3. 结论（算力判据）")
    ok = True

    tf_bf16 = bench_r.get("bf16_tflops")
    tf_fp32 = bench_r.get("fp32_tflops")
    sm = env.get("sm_count")
    clk = bench_r.get("sm_clock_load_mhz")

    if not bench_r:
        print("  [SKIP] 没跑基准测试，无法判定算力")
        bench_r["bench_ok"] = None
        return ok

    if tf_bf16 is None or tf_fp32 is None:
        print("  [FAIL] 基准测试没产出有效数值（计时为负/零）—— 判据无法判定")
        bench_r["bench_ok"] = False
        return False

    # --- 判据 A：Tensor Core 是否生效（与型号无关）
    ratio = tf_bf16 / tf_fp32
    print("  bf16 %.1f / fp32 %.1f = %.2fx   (判据 >= %.1f 才算走了 Tensor Core)"
          % (tf_bf16, tf_fp32, ratio, RATIO_MIN))
    if ratio < RATIO_MIN:
        print("  [FAIL] 比值 %.2fx < %.1f —— Tensor Core 没被用上（降到纯 SIMT 了）"
              % (ratio, RATIO_MIN))
        ok = False
    else:
        print("  [OK]   比值 %.2fx -> Tensor Core 确实在工作" % ratio)

    # --- 判据 B：算力是否打满（分母 = 实测持续时钟算出的上限）
    if clk:
        peak = peak_tflops(sm, clk)
        bench_r["peak_bf16_tflops_at_load_clk"] = round(peak, 1)
        pct = tf_bf16 / peak * 100.0
        bench_r["bf16_pct_of_peak"] = round(pct, 1)
        print("  达成率 : %.1f TFLOPS / %.1f TFLOPS = %.1f%%"
              % (tf_bf16, peak, pct))
        print("           （分母 = %s SM x 128 core x 2 x 2 x 实测 %.0f MHz）"
              % (sm, clk))
        if pct < PEAK_PCT_WARN:
            print("  [WARN] 低于 %.0f%% —— kernel 没打满，或时钟被压（看下面功耗）"
                  % PEAK_PCT_WARN)
        else:
            print("  [OK]   已打满实测时钟下的算力上限")
    else:
        print("  [SKIP] 没采到时钟，跳过达成率判定")

    # --- 判据 C：是否撞功耗墙（解释了时钟为什么低于芯片上限）
    pl, lm, cm = bench_r.get("power_load_w"), bench_r.get("power_limit_w"), env.get("sm_clock_max_mhz")
    if pl and lm:
        pct_p = pl / lm * 100
        print("  功耗   : %.1f W / %.0f W = %.0f%%" % (pl, lm, pct_p))
        if pct_p > 95:
            print("  [NOTE] 已顶到功耗墙 —— 这是**正常且预期**的：180W 卡的物理上限。")
        if cm and clk:
            print("  [NOTE] 持续时钟 %.0f MHz vs 芯片上限 %.0f MHz (%.0f%%)："
                  "差额就是功耗墙导致的降频，不是故障"
                  % (clk, cm, clk / cm * 100))

    drift = bench_r.get("bf16_drift_pct")
    tf_fp16 = bench_r.get("fp16_tflops")
    if drift is not None and tf_fp16 is not None:
        if drift < -10:
            print("  [WARN] bf16 复测比首轮低 %.1f%% -> 存在持续负载降频" % -drift)
        if tf_fp16 < tf_bf16 * 0.9:
            print("  [NOTE] fp16 (%.1f) 明显低于 bf16 (%.1f)：两者理论峰值相同，"
                  "通常是 cuBLAS 给 fp16 选了不同 kernel；重跑一次若消失即为一次性抖动"
                  % (tf_fp16, tf_bf16))

    bench_r["bench_ok"] = ok
    return ok


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bench", action="store_true",
                    help="只验证环境，跳过基准测试（不占显存，随时可跑）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="baseline json 输出路径")
    args = ap.parse_args()

    print("mllm-lab 环境验证  %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("host: %s   %s" % (platform.node(), platform.platform()))

    env, fails = check_env()
    bench_r = {}
    if not args.no_bench and env.get("cuda_available"):
        bench_r = bench()
        if not verdict(env, bench_r):
            fails.append("算力判据未通过（见上面第三节）")

    sec("4. 总判定")
    if fails:
        print("  [FAIL] %d 项未通过：" % len(fails))
        for msg in fails:
            print("    - %s" % msg)
    else:
        print("  [OK]   环境验证全部通过")

    record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "device": env.get("device"),
        "sm_count": env.get("sm_count"),
        "sm_clock_max_mhz": env.get("sm_clock_max_mhz"),
        "driver": env.get("driver"),
        "nvidia_smi_raw": env.get("nvidia_smi_raw"),
        "python": env.get("python"),
        "python_exe": env.get("python_exe"),
        "torch": env.get("torch"),
        "cudnn": env.get("cudnn"),
        "cuda": env.get("cuda"),
        "capability": env.get("capability"),
        "arch_list": env.get("arch_list"),
        "mem_total_mib": env.get("mem_total_mib"),
        "mem_free_mib": env.get("mem_free_mib"),
        "mem_desktop_mib": env.get("mem_desktop_mib"),
        "env_check_passed": not fails,
        "env_check_failures": fails,
    }
    record.update(bench_r)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print("\n  已写入 %s" % args.out)

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
