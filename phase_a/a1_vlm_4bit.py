#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A1 验收 · Qwen2.5-VL-3B-Instruct 4bit(NF4) 推理冒烟测试
=====================================================
目标：在这台 RTX 5060 Ti 8GB / sm_120 上跑通一次**真正的** VLM 推理，
      并记录两个基线数字：**峰值显存** 与 **首 token 延迟 (TTFT)**。

产出：phase_a/a1_vlm_result.json

用法
----
    conda activate mllm
    cd ~/mllm-lab
    python phase_a/a1_vlm_4bit.py                 # 默认 4bit NF4 + 合成测试图
    python phase_a/a1_vlm_4bit.py --repeat 2      # 跑两遍，报告以第二遍为准（第一遍含冷启动）
    python phase_a/a1_vlm_4bit.py --image foo.jpg # 换成自己的图
    A1_ONLINE=1 python phase_a/a1_vlm_4bit.py     # 允许联网（默认强制离线）

设计说明（都是踩过坑的地方）
--------------------------
* 全部计时用 time.perf_counter()（单调时钟）。绝不用 time.time()——WSL2 是虚拟机，
  宿主 NTP 校正会让时钟回跳，实测出现过 -17.15 ms 的负耗时。
* 先做 5 秒的「4bit kernel 预检」再加载 3GB 模型：如果 sm_120 的 kernel 有问题，
  在预检阶段就报出来，不用等模型加载完才发现。
* 测试图是**本地合成**的，不依赖网络，形状/颜色/文字已知，方便判断模型输出对不对。
* max_pixels 显式限制到 0.4 MP —— 8G 下「动态分辨率」是隐形 OOM 杀手。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- 路径与常量
ROOT = pathlib.Path(__file__).resolve().parent          # .../mllm-lab/phase_a
ASSETS = ROOT / "assets"
MODEL_DIR = pathlib.Path.home() / "models" / "Qwen2.5-VL-3B-Instruct"
RESULT_PATH = ROOT / "a1_vlm_result.json"

DEFAULT_MIN_PIXELS = 256 * 28 * 28          # ≈0.20 MP
DEFAULT_MAX_PIXELS = 512 * 28 * 28          # ≈0.40 MP

PROMPT = (
    "看图回答，只输出事实，不要客套。\n"
    "1) 图里有哪几种形状？分别是什么颜色？\n"
    "2) 从上到下逐行读出图中的文字。\n"
    "3) 三个形状里，哪一个在图片最右边？"
)

EXIT_OK, EXIT_ENV, EXIT_OOM, EXIT_OTHER = 0, 2, 3, 1


def setup_env(online: bool) -> None:
    """必须在 import torch 之前调用。"""
    # 显存碎片化：8G 卡上必开
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if online:
        # 国内直连 HF 基本会超时，走镜像
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    else:
        # 权重已在本地，禁止任何联网探测 —— 否则会卡在连接超时上（WSL 内默认无代理）
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def rule(title: str = "") -> None:
    print("\n" + "=" * 68)
    if title:
        print(title)
        print("=" * 68)


def smi(query: str):
    """读 nvidia-smi；WSL 里可用，沙箱里被拉黑时静默返回 None。"""
    try:
        p = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------- 测试图
def make_test_image(path: pathlib.Path) -> pathlib.Path:
    """合成一张内容已知的测试图（不依赖网络，可复现）。"""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 720, 460
    img = Image.new("RGB", (W, H), (248, 248, 245))
    d = ImageDraw.Draw(img)

    d.rectangle([40, 40, 220, 220], fill=(214, 60, 58))                 # 左：红方块
    d.ellipse([280, 60, 420, 200], fill=(58, 108, 214))                 # 中：蓝圆
    d.polygon([(560, 200), (650, 60), (690, 200)], fill=(42, 158, 92))  # 右：绿三角

    try:                                    # Pillow >= 10.1 才支持 size 参数
        font = ImageFont.load_default(size=30)
    except TypeError:
        font = ImageFont.load_default()

    for i, line in enumerate(["MLLM LAB 2026", "square = red",
                              "circle = blue", "triangle = green"]):
        d.text((40, 270 + i * 44), line, fill=(25, 25, 25), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------- 预检
def preflight() -> dict:
    """① 环境断言 ② 5 秒内验证 sm_120 上的 4bit kernel 是否真的能跑。"""
    import torch

    info: dict = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    }

    print(f"torch        : {info['torch']}")
    print(f"cuda         : {info['cuda_available']}")
    print(f"device       : {info['device']}  cap={info['capability']}")
    print(f"arch_list    : {info['arch_list']}")

    if not info["cuda_available"]:
        print("\n[FAIL] CUDA 不可用 —— 这套环境没有真正跑在 GPU 上。")
        sys.exit(EXIT_ENV)
    if "sm_120" not in info["arch_list"]:
        print("\n[FAIL] arch_list 里没有 sm_120 —— torch wheel 不含 Blackwell kernel。")
        sys.exit(EXIT_ENV)

    try:
        import bitsandbytes as bnb
        from bitsandbytes import functional as bf
    except Exception as e:                                  # noqa: BLE001
        print(f"\n[FAIL] bitsandbytes 导入失败：{e!r}")
        print("       → pip install -U bitsandbytes")
        sys.exit(EXIT_ENV)

    info["bitsandbytes"] = bnb.__version__
    print(f"bitsandbytes : {bnb.__version__}")

    # ---- NF4 kernel 往返 ----
    # 判据用「相对」误差，不用绝对值：绝对值随数据 scale 变，跨次/跨数据不可比。
    # NF4 的误差上限由它的 level 间距决定，与数据无关：
    #   16 个 level 按标准正态的分位数分布、归一化到 [-1, 1]；
    #   **最外一档的间隔 = 0.3038**（两端不对称，取大的那侧），
    #   故归一化尺度上 |Δ|max ≤ 0.3038/2 ≈ 0.152  ← 这就是该有的量级
    # 直觉：N(0,1) 数据、blocksize=64 时每块 absmax≈2.5（全局最大可到 ~4.0），
    #       0.13 × 4.0 ≈ 0.5 —— 所以**绝对误差 0.5 是正常的，不是 bug**。
    # 2026-09-23 本地用同一张 NF4 码本离线复算（n=16384 / bs=64）：
    #   |Δ|max = 0.492、相对最大误差 0.122、相对 RMSE 0.092
    #   —— 与实测 0.5 一致，判据随即改为相对量。
    try:
        torch.manual_seed(0)
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        q, st = bf.quantize_4bit(x, quant_type="nf4")
        y = bf.dequantize_4bit(q, st).to(torch.float32)
        xf = x.to(torch.float32)
        d = (xf - y).abs()
        rel_max = (d.max() / xf.abs().max()).item()
        rel_rmse = (d.pow(2).mean().sqrt() / xf.std()).item()

        info.update({
            "nf4_abs_max_err": d.max().item(),
            "nf4_rel_max_err": rel_max,
            "nf4_rel_rmse": rel_rmse,
        })
        print("4bit kernel  : OK（NF4 → 反量化往返）")
        print(f"  |Δ|max      : {d.max().item():.3f}   ← 绝对值随数据 scale 变，别拿它当判据")
        print(f"  相对最大误差 : {rel_max:.3f}   ← 判据在这：理论上限 0.152，实测典型 0.11–0.13")
        print(f"  相对 RMSE   : {rel_rmse:.4f}   ← 实测典型 0.09")
        if rel_max < 0.03:
            print("  [WARN] 相对误差几乎为 0 —— 可疑：量化可能根本没生效")
        elif rel_max > 0.25:
            print("  [WARN] 相对误差超出理论上限（0.152）—— 数值通路有问题")
        del x, q, y, xf, d
        torch.cuda.empty_cache()
    except Exception as e:                                  # noqa: BLE001
        print(f"\n[FAIL] 4bit kernel 跑不起来：{e!r}")
        print("       这就是 sm_120 的 kernel 没编进 wheel —— 换新版 bitsandbytes：")
        print("       pip install -U --force-reinstall bitsandbytes")
        sys.exit(EXIT_ENV)

    return info


# ---------------------------------------------------------------- 加载
def load_model(min_pixels: int, max_pixels: int):
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"model_dir    : {MODEL_DIR}")
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},             # 全部放 GPU0（4bit 只有 ~2GB，不需要 offload）
        attn_implementation="sdpa",     # 不依赖 flash-attn
        low_cpu_mem_usage=True,
    )
    t_load = time.perf_counter() - t0
    model.eval()

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR), min_pixels=min_pixels, max_pixels=max_pixels
    )
    print(f"load_time    : {t_load:.1f} s")
    return model, processor, t_load


# ---------------------------------------------------------------- 单次推理
def run_once(model, processor, image_path: pathlib.Path, max_new_tokens: int) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import TextIteratorStreamer

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    th = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    th.start()

    chunks, t_ttft = [], None
    for chunk in streamer:                      # 阻塞直到有 token 吐出
        if t_ttft is None and chunk.strip():
            t_ttft = time.perf_counter() - t0   # 首个**有效** token
        chunks.append(chunk)
    th.join()
    t_total = time.perf_counter() - t0

    out = "".join(chunks)
    n_out = len(processor.tokenizer(out, add_special_tokens=False)["input_ids"])

    decode_tps = None
    if t_ttft and n_out > 1 and t_total > t_ttft:
        decode_tps = (n_out - 1) / (t_total - t_ttft)

    return {
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": n_out,
        "ttft_s": t_ttft,
        "total_s": t_total,
        "decode_tok_per_s": decode_tps,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "answer": out.strip(),
    }


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen2.5-VL-3B 4bit 推理冒烟测试")
    ap.add_argument("--image", type=str, default=None, help="自定义图片；默认用合成图")
    ap.add_argument("--repeat", type=int, default=1, help="重复次数，报告以最后一次为准")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--online", action="store_true", help="允许联网（默认离线）")
    args = ap.parse_args()

    setup_env(online=args.online or os.environ.get("A1_ONLINE") == "1")

    result: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}
    result["gpu_mem_used_before_mib"] = smi("memory.used")
    print(f"gpu mem used (before) : {result['gpu_mem_used_before_mib']} MiB")

    rule("① 预检")
    result["env"] = preflight()

    rule("② 准备测试图")
    image_path = pathlib.Path(args.image) if args.image else ASSETS / "test_card.png"
    if args.image is None:
        make_test_image(image_path)
    print(f"image        : {image_path}")
    result["image"] = str(image_path)

    rule("③ 加载模型（4bit NF4）")
    try:
        model, processor, t_load = load_model(args.min_pixels, args.max_pixels)
        result["load_time_s"] = t_load
    except Exception as e:                                   # noqa: BLE001
        if is_oom(e):
            print("\n[FAIL] 加载阶段 OOM —— 先把桌面程序（壁纸/浏览器/Zotero）关掉，再看 nvidia-smi")
            return EXIT_OOM
        print(f"\n[FAIL] 加载失败：{e!r}")
        print("       把这段报错发我，不要自己硬试。")
        return EXIT_OTHER

    rule("④ 推理")
    runs = []
    for i in range(max(1, args.repeat)):
        print(f"--- run {i + 1}/{max(1, args.repeat)} ---")
        try:
            r = run_once(model, processor, image_path, args.max_new_tokens)
        except Exception as e:                               # noqa: BLE001
            if is_oom(e):
                print("\n[FAIL] 推理阶段 OOM —— 调小 --max-pixels 或 --max-new-tokens")
                return EXIT_OOM
            print(f"\n[FAIL] 推理失败：{e!r}")
            return EXIT_OTHER
        runs.append(r)
        tps = r["decode_tok_per_s"]
        print(f"  TTFT        : {r['ttft_s']:.3f} s")
        print(f"  total       : {r['total_s']:.2f} s  ({r['output_tokens']} tok, "
              f"decode {tps:.1f} tok/s)" if tps else
              f"  total       : {r['total_s']:.2f} s  ({r['output_tokens']} tok)")
        print(f"  peak mem    : {r['peak_allocated_mib']:.0f} MiB allocated / "
              f"{r['peak_reserved_mib']:.0f} MiB reserved")
        print(f"  answer      :\n{r['answer']}\n")

    result["runs"] = runs
    last = runs[-1]
    result["report"] = {
        "ttft_s": last["ttft_s"],
        "decode_tok_per_s": last["decode_tok_per_s"],
        "peak_allocated_mib": last["peak_allocated_mib"],
        "peak_reserved_mib": last["peak_reserved_mib"],
        "output_tokens": last["output_tokens"],
    }
    result["gpu_mem_used_after_mib"] = smi("memory.used")

    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    rule("⑤ 结论")
    tps = last["decode_tok_per_s"]
    print(f"TTFT                 : {last['ttft_s']:.3f} s" if last["ttft_s"] else "TTFT                 : 无（没吐出 token）")
    print(f"decode 速度          : {tps:.1f} tok/s" if tps else "decode 速度          : 无")
    print(f"峰值显存 (torch)     : {last['peak_allocated_mib']:.0f} MiB")
    print(f"峰值显存 (reserved)  : {last['peak_reserved_mib']:.0f} MiB")
    print(f"nvidia-smi 占用      : {result['gpu_mem_used_before_mib']} → "
          f"{result['gpu_mem_used_after_mib']} MiB")
    print(f"\n结果已写入 {RESULT_PATH}")
    print("→ 把 TTFT / decode 速度 / 峰值显存 三个数抄进台账「环境状态」表。")
    return EXIT_OK


def is_oom(e: BaseException) -> bool:
    """判断是不是显存不足（torch 的 OOM 类名跨版本有过变化，用字符串兜底）。"""
    return "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n手动中断。")
        sys.exit(130)
