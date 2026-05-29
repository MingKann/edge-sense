"""
阶段1：采集 + 推理原型
────────────────────────
摄像头拍一帧 → 提取基础统计 → 拼成文本prompt → LLM给一句文字回复

这是整个项目的最小闭环，验证"视觉信号→结构化文本→LLM语义推理"管线可用。
"""

import sys
import time
import numpy as np
from camera import Camera
from inference import OllamaInference

# ── 配置 ──────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MODEL_NAME = "vantage"
OLLAMA_URL = "http://localhost:11434"

# ── System Prompt ─────────────────────────────────────
# 阶段1只需要简单角色设定，阶段3会深度设计
SYSTEM_PROMPT = (
    "你是一个嵌入式视觉诊断助手。"
    "你会收到一张图像的统计特征数据（分辨率、亮度、颜色分布等）。"
    "请基于这些数据，用一句话描述画面中可能是什么场景，并给出你的置信度(0-100%)。"
    "只输出一句话，不要额外解释。"
)

# ── 预处理：图像 → 统计文本 ──────────────────────────
def extract_frame_stats(frame: np.ndarray) -> dict:
    """
    从一帧图像中提取基础统计特征。

    返回 dict:
        - width, height: 分辨率
        - b_mean, g_mean, r_mean: BGR通道均值 (float)
        - brightness_mean: 平均亮度 (float, 0-255)
        - brightness_p10/p50/p90: 亮度分位数
        - dominant_color: 主导色调 (str: "偏蓝"/"偏绿"/"偏红"/"偏暗"/"中性")
    """
    h, w = frame.shape[:2]
    # 注意：上面用了 cv2.split，但本文件不直接 import cv2，由 camera 模块隐式依赖
    # 为清晰起见，直接在函数内计算
    b_arr = frame[:, :, 0].astype(np.float64)
    g_arr = frame[:, :, 1].astype(np.float64)
    r_arr = frame[:, :, 2].astype(np.float64)

    # RGB 均值
    b_mean = float(b_arr.mean())
    g_mean = float(g_arr.mean())
    r_mean = float(r_arr.mean())

    # 亮度 = 0.299*R + 0.587*G + 0.114*B (BT.601)
    brightness = 0.299 * r_arr + 0.587 * g_arr + 0.114 * b_arr
    brightness_flat = brightness.flatten()
    brightness_flat.sort()  # 排序后取分位数

    n = len(brightness_flat)
    p10 = float(brightness_flat[int(n * 0.10)])
    p50 = float(brightness_flat[int(n * 0.50)])
    p90 = float(brightness_flat[int(n * 0.90)])
    brightness_mean = float(brightness.mean())

    # 主导色调
    means = {"蓝": b_mean, "绿": g_mean, "红": r_mean}
    dominant = max(means, key=means.get)  # "蓝"/"绿"/"红"
    max_val = means[dominant]
    min_val = min(means.values())
    if max_val - min_val < 10:
        dominant_str = "中性（无明显偏色）"
    elif brightness_mean < 50:
        dominant_str = "偏暗"
    else:
        dominant_str = f"偏{dominant}"

    return {
        "width": w,
        "height": h,
        "b_mean": round(b_mean, 1),
        "g_mean": round(g_mean, 1),
        "r_mean": round(r_mean, 1),
        "brightness_mean": round(brightness_mean, 1),
        "brightness_p10": round(p10, 1),
        "brightness_p50": round(p50, 1),
        "brightness_p90": round(p90, 1),
        "dominant_color": dominant_str,
    }


def stats_to_prompt(stats: dict) -> str:
    """将统计 dict 转为 LLM 可理解的文本 prompt"""
    return (
        f"图像分辨率: {stats['width']}×{stats['height']}\n"
        f"BGR均值: B={stats['b_mean']}, G={stats['g_mean']}, R={stats['r_mean']}\n"
        f"亮度均值: {stats['brightness_mean']}/255\n"
        f"亮度分位数(10%/50%/90%): {stats['brightness_p10']}/{stats['brightness_p50']}/{stats['brightness_p90']}\n"
        f"主导色调: {stats['dominant_color']}\n"
        f"\n请描述画面中可能是什么场景。"
    )


# ── 主流程 ─────────────────────────────────────────────
def main():
    import cv2  # 仅此处导入，避免 camera.py 遗漏依赖时报错不清晰

    print("=" * 50)
    print(" Vantage · 阶段1：采集 + 推理原型")
    print("=" * 50)

    # 1. 初始化推理引擎
    llm = OllamaInference(model=MODEL_NAME, base_url=OLLAMA_URL)

    print("\n[1/4] 检测 Ollama 服务...")
    if not llm.ping():
        print("❌ Ollama 未运行！请确认托盘图标或手动启动 ollama serve")
        sys.exit(1)
    print(f"✅ Ollama 在线，模型={MODEL_NAME}")

    # 2. 打开摄像头
    print("\n[2/4] 打开摄像头...")
    with Camera(index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT) as cam:
        # 预热：丢弃前几帧（摄像头自动曝光/白平衡稳定）
        print("    预热中（丢弃前5帧）...")
        for _ in range(5):
            cam.capture()
            time.sleep(0.05)

        # 3. 捕获一帧 + 提取统计
        print("\n[3/4] 捕获一帧 + 提取统计特征...")
        frame = cam.capture()
        stats = extract_frame_stats(frame)
        print(f"    统计结果: {stats}")

        # 4. 推理
        print("\n[4/4] 发送 prompt 到 LLM 推理...")
        prompt = stats_to_prompt(stats)
        print(f"    Prompt:\n{prompt}\n")

        t0 = time.time()
        response = llm.generate(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.7)
        elapsed = time.time() - t0

        print(f"    ⏱ 推理耗时: {elapsed:.1f}秒")
        print(f"\n{'─' * 40}")
        print(f" 🤖 LLM 回复: {response}")
        print(f"{'─' * 40}")

    print("\n✅ 阶段1原型验证完成")


if __name__ == "__main__":
    main()
