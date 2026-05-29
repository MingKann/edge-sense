"""
阶段2：预处理管线测试
───────────────────────
连续采集N帧 → 每帧调用 analyze_frame() → 打印结构化诊断结果。
不调用 LLM（那是阶段3的事），仅验证四模块产出数据合理性。
"""

import sys
import time
import cv2
import yaml

from camera import Camera
from preprocess import FrameAnalyzer


def load_config(path: str = "src/config.yaml") -> dict:
    """加载 YAML 配置"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        print(f"⚠️ 无法加载 {path}，使用默认配置")
        return {}


def main():
    cfg = load_config("src/config.yaml")

    print("=" * 55)
    print(" Vantage · 阶段2：预处理管线测试")
    print("=" * 55)

    # 初始化
    analyzer = FrameAnalyzer(cfg.get("preprocess", {}))
    N_FRAMES = 200  # 采集200帧（约6-7秒）

    print(f"\n采集 {N_FRAMES} 帧，请将摄像头对准设备面板...")
    print("（MOG2 前30帧为背景学习期，FFT 前128帧为缓冲区填充期）\n")

    with Camera(
        index=cfg.get("camera", {}).get("index", 0),
        width=cfg.get("camera", {}).get("width", 640),
        height=cfg.get("camera", {}).get("height", 480),
    ) as cam:
        # 预热
        for _ in range(5):
            cam.capture()
            time.sleep(0.03)

        for i in range(N_FRAMES):
            frame = cam.capture()
            result = analyzer.analyze_frame(frame)

            # 简略打印（避免刷屏）
            c = result["color"]
            m = result["motion"]
            f = result["flicker"]
            o = result["ocr"]

            print(
                f"[{result['frame_id']:03d}] "
                f"色温={c['color_temperature']:7s} "
                f"运动={m['level']:6s}({m['motion_ratio']:.4f}) "
                f"闪烁={f['status']:13s} freq={f['frequency_hz']:5.1f}Hz "
                f"OCR={o['status']:13s} text={o['text'][:30] if o['text'] else '-'}"
            )

    print("\n✅ 阶段2管线测试完成")


if __name__ == "__main__":
    main()
