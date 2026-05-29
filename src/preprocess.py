"""
预处理管线模块 (阶段2)
───────────────────────
FrameAnalyzer 类：维护跨帧状态，将原始帧转化为结构化诊断数据。

四模块：
  - 颜色检测: K-means 聚类 → 主色 + 色温
  - 运动检测: MOG2 背景建模 → 运动强度 + 区域
  - 频率分析: ROI 亮度序列 → FFT → LED 闪烁频率
  - OCR读数: Tesseract → 面板数字/文字提取
"""

import time
import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

# Tesseract OCR 路径
# 优先级: TESSERACT_CMD 环境变量 → Windows 用户级安装 → 系统 PATH
import os
import pytesseract

_tesseract_env = os.environ.get("TESSERACT_CMD")
if _tesseract_env:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_env
else:
    _win_path = r"C:\Users\mingkann\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win_path):
        pytesseract.pytesseract.tesseract_cmd = _win_path


# ── 数据类：各模块返回值 ──────────────────────────────────

@dataclass
class ColorResult:
    dominant_colors: List[dict] = field(default_factory=list)
    mean_hue: float = 0.0
    mean_saturation: float = 0.0
    mean_value: float = 0.0
    color_temperature: str = "neutral"
    is_monochrome: bool = False


@dataclass
class MotionResult:
    status: str = "initializing"  # initializing / ready
    level: str = "none"           # none / low / medium / high
    motion_ratio: float = 0.0
    num_regions: int = 0
    largest_region_area: int = 0


@dataclass
class FlickerResult:
    status: str = "initializing"  # initializing / ready / no_led_found
    led_roi: Optional[Tuple[int, int, int, int]] = None
    frequency_hz: float = 0.0
    amplitude: float = 0.0
    is_stable: bool = True


@dataclass
class OCRResult:
    status: str = "unavailable"   # ok / unavailable / no_text_found
    text: str = ""
    confidence: float = 0.0
    numeric_value: Optional[float] = None


# ── FrameAnalyzer ────────────────────────────────────────

class FrameAnalyzer:
    """
    帧分析器 — 维护跨帧状态，逐帧输出结构化诊断。

    使用方式:
        analyzer = FrameAnalyzer(config)
        for frame in camera_stream:
            result = analyzer.analyze_frame(frame)
            print(result)  # dict
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        # ── 颜色检测配置 ──
        color_cfg = cfg.get("color", {})
        self._k_colors: int = color_cfg.get("k_colors", 5)
        self._color_sample_step: int = color_cfg.get("sample_step", 4)

        # ── 运动检测配置 ──
        motion_cfg = cfg.get("motion", {})
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=motion_cfg.get("history", 100),
            varThreshold=motion_cfg.get("var_threshold", 36),
            detectShadows=False,
        )
        self._motion_low_thresh: float = motion_cfg.get("low_threshold", 0.005)
        self._motion_medium_thresh: float = motion_cfg.get("medium_threshold", 0.03)
        self._motion_high_thresh: float = motion_cfg.get("high_threshold", 0.10)
        self._motion_min_area: int = motion_cfg.get("min_area", 200)
        self._motion_warmup_frames: int = motion_cfg.get("warmup_frames", 30)
        self._frame_count: int = 0

        # ── 频率分析配置 ──
        flicker_cfg = cfg.get("flicker", {})
        self._flicker_buffer_size: int = flicker_cfg.get("buffer_size", 128)
        self._flicker_brightness_history: deque = deque(maxlen=self._flicker_buffer_size)
        self._flicker_led_roi: Optional[Tuple[int, int, int, int]] = None
        roi_cfg = flicker_cfg.get("roi", None)
        if roi_cfg:
            self._flicker_led_roi = (
                roi_cfg.get("x", 0), roi_cfg.get("y", 0),
                roi_cfg.get("w", 32), roi_cfg.get("h", 32),
            )
        self._flicker_led_threshold: int = flicker_cfg.get("led_threshold", 180)
        self._flicker_led_min_area: int = flicker_cfg.get("led_min_area", 4)
        self._flicker_led_max_area: int = flicker_cfg.get("led_max_area", 400)
        self._flicker_ready: bool = False
        self._flicker_roi_size: int = flicker_cfg.get("roi_size", 24)
        self._flicker_fps: float = flicker_cfg.get("fps", 30.0)

        # ── OCR 配置 ──
        ocr_cfg = cfg.get("ocr", {})
        self._ocr_enabled: bool = ocr_cfg.get("enabled", True)
        self._ocr_interval: int = ocr_cfg.get("interval", 10)
        self._ocr_whitelist: str = ocr_cfg.get(
            "whitelist", "0123456789.-:AVWHzFkMmμ°CΩ"
        )
        self._ocr_psm: int = ocr_cfg.get("psm", 7)
        self._ocr_last_result: OCRResult = OCRResult()
        self._ocr_available: Optional[bool] = None

        # ── 帧计数 ──
        self._frame_id: int = 0

    # ── 公开接口 ─────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray) -> dict:
        self._frame_id += 1
        self._frame_count += 1
        t0 = time.time()

        color = self._analyze_colors(frame)
        motion = self._analyze_motion(frame)
        flicker = self._analyze_flicker(frame)
        ocr = self._analyze_ocr(frame)

        return {
            "frame_id": self._frame_id,
            "timestamp": t0,
            "color": color.__dict__,
            "motion": motion.__dict__,
            "flicker": flicker.__dict__,
            "ocr": ocr.__dict__,
        }

    # ── 颜色检测 ─────────────────────────────────────────

    def _analyze_colors(self, frame: np.ndarray) -> ColorResult:
        result = ColorResult()
        h, w = frame.shape[:2]
        small = frame[::self._color_sample_step, ::self._color_sample_step]
        pixels = small.reshape(-1, 3).astype(np.float32)

        if len(pixels) < self._k_colors:
            return result

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, self._k_colors, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = centers.astype(np.uint8)

        label_counts = np.bincount(labels.flatten(), minlength=self._k_colors)
        total = label_counts.sum()

        color_labels = self._label_colors(centers)
        for i in np.argsort(label_counts)[::-1]:
            ratio = float(label_counts[i] / total)
            if ratio < 0.03:
                continue
            b, g, r = centers[i].tolist()
            result.dominant_colors.append({
                "hex": f"#{r:02X}{g:02X}{b:02X}",
                "rgb": [r, g, b],
                "ratio": round(ratio, 4),
                "label": color_labels[i],
            })

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result.mean_hue = round(float(hsv[:, :, 0].mean()), 1)
        result.mean_saturation = round(float(hsv[:, :, 1].mean()), 1)
        result.mean_value = round(float(hsv[:, :, 2].mean()), 1)

        if result.mean_saturation < 20:
            result.color_temperature = "neutral"
        elif result.mean_hue < 20 or result.mean_hue > 160:
            result.color_temperature = "warm"
        elif 80 < result.mean_hue < 140:
            result.color_temperature = "cool"
        else:
            result.color_temperature = "neutral"

        result.is_monochrome = result.mean_saturation < 15
        return result

    @staticmethod
    def _label_colors(centers: np.ndarray) -> List[str]:
        labels = []
        for bgr in centers:
            b, g, r = bgr.astype(int)
            if max(b, g, r) - min(b, g, r) < 25 and max(b, g, r) < 50:
                labels.append("黑")
            elif max(b, g, r) - min(b, g, r) < 25 and max(b, g, r) > 200:
                labels.append("白")
            elif max(b, g, r) - min(b, g, r) < 30:
                labels.append("灰")
            elif r > g and r > b:
                labels.append("红" if r - max(g, b) > 50 else "橙")
            elif g > r and g > b:
                labels.append("绿")
            elif b > r and b > g:
                labels.append("蓝")
            elif r > 120 and g > 100 and b < 80:
                labels.append("黄")
            else:
                labels.append("其他")
        return labels

    # ── 运动检测 ─────────────────────────────────────────

    def _analyze_motion(self, frame: np.ndarray) -> MotionResult:
        result = MotionResult()

        if self._frame_count < self._motion_warmup_frames:
            self._mog2.apply(frame)
            result.status = "initializing"
            return result

        result.status = "ready"
        fg_mask = self._mog2.apply(frame, learningRate=0.005)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        motion_pixels = int(cv2.countNonZero(fg_mask))
        total_pixels = fg_mask.size
        result.motion_ratio = round(motion_pixels / total_pixels, 6)

        if result.motion_ratio < self._motion_low_thresh:
            result.level = "none"
        elif result.motion_ratio < self._motion_medium_thresh:
            result.level = "low"
        elif result.motion_ratio < self._motion_high_thresh:
            result.level = "medium"
        else:
            result.level = "high"

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self._motion_min_area]
        result.num_regions = len(valid_contours)

        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            result.largest_region_area = int(cv2.contourArea(largest))

        return result

    # ── 频率分析 ─────────────────────────────────────────

    def _analyze_flicker(self, frame: np.ndarray) -> FlickerResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._flicker_led_roi is None:
            self._flicker_led_roi = self._detect_led_roi(gray)

        result = FlickerResult()

        if self._flicker_led_roi is None:
            result.status = "no_led_found"
            return result

        x, y, w, h = self._flicker_led_roi
        result.led_roi = (x, y, w, h)

        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            result.status = "no_led_found"
            return result

        mean_brightness = float(roi.mean())
        self._flicker_brightness_history.append(mean_brightness)

        if len(self._flicker_brightness_history) < self._flicker_buffer_size:
            result.status = "initializing"
            return result

        result.status = "ready"
        signal = np.array(self._flicker_brightness_history, dtype=np.float64)
        signal = signal - signal.mean()
        result.amplitude = round(float(signal.std()), 2)

        if result.amplitude < 1.5:
            result.is_stable = True
            result.frequency_hz = 0.0
            return result

        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self._flicker_fps)
        magnitude = np.abs(fft)

        min_freq_idx = max(1, int(0.5 / (self._flicker_fps / len(signal))))
        if min_freq_idx >= len(magnitude):
            result.is_stable = True
            return result

        peak_idx = min_freq_idx + np.argmax(magnitude[min_freq_idx:])
        result.frequency_hz = round(float(freqs[peak_idx]), 2)
        result.is_stable = result.frequency_hz < 0.5 or result.amplitude < 2.0

        return result

    def _detect_led_roi(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        _, thresh = cv2.threshold(gray, self._flicker_led_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self._flicker_led_min_area <= area <= self._flicker_led_max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                candidates.append((x, y, cw, ch))

        if not candidates:
            return None

        best = max(candidates, key=lambda r: float(gray[r[1]:r[1]+r[3], r[0]:r[0]+r[2]].mean()))
        cx = best[0] + best[2] // 2
        cy = best[1] + best[3] // 2
        half = self._flicker_roi_size // 2
        x = max(0, cx - half)
        y = max(0, cy - half)
        w = h = min(self._flicker_roi_size, gray.shape[1] - x, gray.shape[0] - y)

        return (x, y, w, h)

    # ── OCR 读数 ─────────────────────────────────────────

    def _analyze_ocr(self, frame: np.ndarray) -> OCRResult:
        """Tesseract OCR 提取面板数字/文字"""
        if not self._ocr_enabled:
            return OCRResult(status="disabled")

        if self._ocr_available is None:
            self._ocr_available = self._check_tesseract()
            if not self._ocr_available:
                return OCRResult(status="unavailable")

        if self._frame_id % self._ocr_interval != 0:
            return self._ocr_last_result

        try:
            # 预处理：灰度 → CLAHE → 自适应阈值
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 4,
            )

            config = (
                f"--psm {self._ocr_psm}"
                f" -c tessedit_char_whitelist={self._ocr_whitelist}"
            )
            text = pytesseract.image_to_string(binary, config=config).strip()
            data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if c != "-1"]
            avg_conf = float(np.mean(confidences)) / 100.0 if confidences else 0.0

            result = OCRResult(status="ok", text=text, confidence=round(avg_conf, 4))

            if not text:
                result.status = "no_text_found"
            else:
                result.numeric_value = self._parse_numeric(text)

            self._ocr_last_result = result
            return result

        except Exception:
            self._ocr_available = False
            return OCRResult(status="unavailable")

    @staticmethod
    def _check_tesseract() -> bool:
        """检测 Tesseract 是否可用（路径已在模块顶部设置）"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_numeric(text: str) -> Optional[float]:
        import re
        match = re.search(r"-?\d+\.?\d*", text.replace(" ", ""))
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None
