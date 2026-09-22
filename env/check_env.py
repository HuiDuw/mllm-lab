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

--------------------------------------------------------------------------
v2（2026-09-22 实测后修订）
--------------------------------------------------------------------------
v1 用 bf16 < 50 TFLOPS 当 FAIL 判据 —— **这是个错阈值**。
它把一台跑满的 RTX 5060 Ti 误判成了 FAIL（实测 47.9）。

RTX 5060 Ti 的真实上限（GB206 / 36 SM / 4608 core / boost 2.572 GHz）：
    FP32 向量      = 4608 x 2 x 2.572GHz           = 23.7 TFLOPS
    BF16 Tensor    = 2 x FP32                      = 47.4 TFLOPS
    （官方规格页的 "759 AI TOPS" 是 FP4+稀疏口径，/16 得稠密 bf16 = 47.4，一致）

50 卡在 47.4 的上方 —— 阈值比峰值还高，必然误报。

v2 改用**与型号无关的比值判据**：
    bf16_TFLOPS / fp32_TFLOPS >= 2.0
Tensor Core 的稠密 bf16/fp16 吞吐恒等于 FP32 向量吞吐的 2 倍（Ada / Blackwell 消费卡
皆是如此）；若比值 < 2，说明根本没走 Tensor Core（纯 SIMT 的 bf16 只会和 fp32 打平
甚至更慢）。这个判据在任何显卡上都成立，不依赖具体型号的绝对数值。

同时用 SM 数 x 时钟算出理论峰值，报告"实测算力 / 理论峰值"的百分比 —— 用来抓降频。
另加一轮 bf16 复测（放在最后），用来区分「fp16 天生慢」还是「跑热了降频」。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
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
PEAK_PCT_WARN = 70.0           # 实测算力 < 理论峰值 70% = 可能降频/功耗墙

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


def _sm_clock_mhz():
    """最大 SM 时钟（MHz）。取不到返回 None。"""
    raw = _smi("clocks.max.sm", nounits=True)
    try:
        return float(raw.split()[0])
    except Exception:
        return None


def theoretical_peaks(props):
    """(fp32_peak, bf16_peak, sm_count, clk_mhz)，算不出来时对应项为 None。"""
    sm = getattr(props, "multi_processor_count", None)
    clk = _sm_clock_mhz()
    if clk is None:
        khz = getattr(props, "clock_rate", 0) or 0     # torch 新版才有，单位 kHz
        clk = khz / 1000.0 if khz else None
    if not sm or not clk:
        return None, None, sm, clk
    fp32 = sm * FP32_CORES_PER_SM * 2 * clk * 1e6 / 1e12
    return fp32, fp32 * TENSOR_RATIO, sm, clk


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

    # --- 理论峰值（用于后面的「实测算力 / 峰值」百分比）
    fp32_pk, bf16_pk, sm, clk = theoretical_peaks(props)
    r["sm_clock_mhz"] = clk
    r["theoretical_fp32_tflops"] = round(fp32_pk, 1) if fp32_pk else None
    r["theoretical_bf16_tflops"] = round(bf16_pk, 1) if bf16_pk else None
    if bf16_pk:
        print("  理论峰值   : FP32 %.1f / BF16-Tensor %.1f TFLOPS  (%s SM x %s MHz)"
              % (fp32_pk, bf16_pk, sm, clk))
        print("               注：官方规格页的 'AI TOPS' 是 FP4+稀疏口径，除以 16 才可比")
    else:
        print("  理论峰值   : 算不出来（拿不到 SM 时钟）—— 后面只用比值判据")

    return r, fails


# ----------------------------------------------------------------------------
# 2. 基准测试
# ----------------------------------------------------------------------------
def bench_matmul(dtype, n=4096, warmup=5, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt, 2.0 * n ** 3 / dt / 1e12


def bench_membw(nbytes=256 * 1024 * 1024, iters=20):
    """D2D 拷贝。

    ⚠️ 读法：一次拷贝要搬 2N 字节（读 N + 写 N）。
       - copy_gbps    = N/t   「应用层拷贝速率」——只有带宽的一半，别拿去比规格值
       - effective    = 2N/t  「实际带宽」——这才该和 448 GB/s 比（73% 左右正常）
    """
    x = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    gib_per_s = nbytes / 1024 ** 3 / dt
    return dt, gib_per_s, 2 * gib_per_s


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
        dt, tf = bench_matmul(dtype, warmup=warm, iters=it)
        r["%s_ms" % name] = round(dt * 1000, 2)
        r["%s_tflops" % name] = round(tf, 1)
        print("    %.2f ms   %.1f TFLOPS" % (dt * 1000, tf))

    # --- bf16 复测：与首轮对比，用来区分「天生慢」和「跑热了降频」
    print("  bf16 matmul 复测 (检测降频) ...")
    dt, tf = bench_matmul(torch.bfloat16, warmup=5, iters=50)
    r["bf16_repeat_tflops"] = round(tf, 1)
    drift = (tf - r["bf16_tflops"]) / r["bf16_tflops"] * 100.0
    r["bf16_drift_pct"] = round(drift, 1)
    print("    %.2f ms   %.1f TFLOPS   (与首轮差 %+.1f%%)" % (dt * 1000, tf, drift))

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
    """判据全部基于比值 / 百分比，不写死型号绝对数值。"""
    sec("3. 结论（算力判据）")
    ok = True

    tf_bf16 = bench_r.get("bf16_tflops")
    tf_fp32 = bench_r.get("fp32_tflops")
    peak = env.get("theoretical_bf16_tflops")

    if tf_bf16 is None or tf_fp32 is None:
        print("  [SKIP] 没跑基准测试，无法判定算力")
        bench_r["bench_ok"] = None
        return ok

    ratio = tf_bf16 / tf_fp32
    print("  bf16 %.1f / fp32 %.1f = %.2fx   (判据 >= %.1f 才算走了 Tensor Core)"
          % (tf_bf16, tf_fp32, ratio, RATIO_MIN))
    if ratio < RATIO_MIN:
        print("  [FAIL] 比值 %.2fx < %.1f —— Tensor Core 没被用上（降到纯 SIMT 了）"
              % (ratio, RATIO_MIN))
        ok = False
    else:
        print("  [OK]   比值 %.2fx -> Tensor Core 确实在工作" % ratio)

    if peak:
        pct = tf_bf16 / peak * 100.0
        bench_r["bf16_pct_of_peak"] = round(pct, 1)
        print("  bf16 实测算力 / 理论峰值 = %.1f%%  (理论 %.1f TFLOPS)"
              % (pct, peak))
        if pct < PEAK_PCT_WARN:
            print("  [WARN] 低于 %.0f%% —— 可能降频 / 撞功耗墙（180W 卡长时间满载会掉）"
                  % PEAK_PCT_WARN)
        else:
            print("  [OK]   接近理论峰值 -> 这张卡跑满了")

    drift = bench_r.get("bf16_drift_pct")
    tf_fp16 = bench_r.get("fp16_tflops")
    if drift is not None and tf_fp16 is not None:
        if drift < -10:
            print("  [WARN] bf16 复测比首轮低 %.1f%% -> 存在持续负载降频，"
                  "前面 fp16 偏慢可能也是这个原因" % -drift)
        if tf_fp16 < tf_bf16 * 0.85:
            print("  [NOTE] fp16 (%.1f) 明显低于 bf16 (%.1f)：本轮没测到降频的话，"
                  "是 cuBLAS 给 fp16 选了不同 kernel —— 正常现象，"
                  "不是硬件问题（两者理论峰值相同）" % (tf_fp16, tf_bf16))

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
        "sm_clock_mhz": env.get("sm_clock_mhz"),
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
        "theoretical_fp32_tflops": env.get("theoretical_fp32_tflops"),
        "theoretical_bf16_tflops": env.get("theoretical_bf16_tflops"),
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
