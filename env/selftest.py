# -*- coding: utf-8 -*-
"""
check_env.py 的离线自测 —— 用 stub torch，**不需要 GPU**，随处可跑。

    python env/selftest.py          # 退出码 0 = 全过

为什么需要它：
    check_env.py 里踩过的坑有两类 —— 一类是"判据取错参考值"（v1/v2/v3），
    另一类是**纯代码 bug**（v4：遮蔽基类方法、用了非单调时钟、返回值元数不匹配）。
    后一类不需要 GPU 就能测出来，所以固化成一个不依赖硬件的冒烟测试。

    这些测试都有明确的"回归"意图，不要为了让它们变绿而改测试。
"""
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class FakeTensor:
    def __matmul__(self, o):
        return self


def _install_stub_torch():
    """必须在 import check_env 之前调用（check_env 顶层就 import torch）。"""
    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(cuda="13.0")
    fake.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: 0))
    fake.cuda = types.SimpleNamespace(empty_cache=lambda: None,
                                      synchronize=lambda: None,
                                      Event=lambda **k: None)
    fake.bfloat16 = "bf16"
    fake.float16 = "fp16"
    fake.float32 = "fp32"
    fake.randn = lambda *a, **k: FakeTensor()
    sys.modules["torch"] = fake


def main():
    _install_stub_torch()
    import check_env

    # --- 回归 1：LoadSampler 不能再遮蔽 threading.Thread._stop()
    print("=== 测试 1: LoadSampler.stop() 不抛 TypeError ===")
    s = check_env.LoadSampler(interval=0.05)
    s.start()
    time.sleep(0.4)
    s.stop()                       # v3 崩在这一行：'Event' object is not callable
    assert not s.is_alive(), "stop() 后线程应已结束"
    print("  [OK] stop() 正常返回，线程已退出")

    # --- 回归 2：负的计时结果必须被丢弃，取正样本中位数
    print("=== 测试 2: bench_matmul 丢弃负值并取中位数 ===")
    seq = iter([0.0030, -0.0009, 0.0031, 0.0029])
    check_env._timed_matmul = lambda a, b, w, i: next(seq)
    dt, tf, spread = check_env.bench_matmul("bf16", n=4096, repeats=4)
    assert dt is not None and dt > 0, "负值必须被丢弃"
    assert abs(dt - 0.0030) < 1e-9, "应取 [0.0030, 0.0031, 0.0029] 的中位数"
    print("  [OK] dt=%.5fs 中位数正确，spread=%.1f%%" % (dt, spread))

    # --- 回归 3：全为负时返回三元组（曾因只返回 2 个值而 ValueError）
    print("=== 测试 3: 全为负 -> 返回 (None, None, None) ===")
    check_env._timed_matmul = lambda a, b, w, i: -0.001
    dt, tf, spread = check_env.bench_matmul("bf16", repeats=3)
    assert dt is None and tf is None and spread is None, "必须是三元组"
    print("  [OK] 三元组返回，调用方可安全解包")

    # --- 回归 4：跑了基准但没数值 -> FAIL，而不是静默 SKIP
    print("=== 测试 4: 无有效数值时判 FAIL ===")
    ok = check_env.verdict({"sm_count": 36}, {"bf16_tflops": None, "fp32_tflops": None})
    assert ok is False, "bench 跑了却没数值 => 必须 FAIL"
    print("  [OK] verdict 返回 False")

    print("\nALL SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
