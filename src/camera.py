"""
摄像头采集模块
- 打开摄像头，设置分辨率
- 捕获单帧（numpy BGR数组）
- 支持 with 语句自动释放
"""

import cv2
import numpy as np
from typing import Tuple


class Camera:
    """USB/内置摄像头封装"""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> None:
        """打开摄像头并设置分辨率"""
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)  # Windows下DSHOW避免黑屏
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 index={self.index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 验证最终分辨率（部分摄像头可能不支持指定分辨率，会自动回退）
        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] 已打开 index={self.index}, 分辨率={int(actual_w)}×{int(actual_h)}")

    def capture(self) -> np.ndarray:
        """捕获一帧，返回 BGR numpy 数组 (H, W, 3)"""
        if self._cap is None:
            raise RuntimeError("摄像头未打开，先调用 open()")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("读取帧失败，摄像头可能断开")
        return frame

    def release(self) -> None:
        """释放摄像头资源"""
        if self._cap is not None:
            self._cap.release()
            print("[Camera] 已释放")

    # --- 上下文管理器协议 ---
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False  # 不吞异常
