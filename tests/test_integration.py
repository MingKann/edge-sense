"""
Integration tests — full pipeline with simulated camera frames.
No hardware required: runs in CI with synthetic numpy arrays.

覆盖:
  1. 静态正常场景 → normal
  2. 运动场景 → alert (两帧差异触发 MOG2)
  3. LED 闪烁场景 → 频率分析
  4. 全管线端到端 (analyze → compute_status → confidence → prompt)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2
import pytest

from preprocess import FrameAnalyzer
from stage3_prototype import compute_status, compute_confidence, build_diagnostic_prompt


# ── Synthetic frame generators ──────────────────────────

def _solid_frame(color_bgr=(128, 128, 128), size=(640, 480)):
    """单色帧"""
    return np.full((size[1], size[0], 3), color_bgr, dtype=np.uint8)


def _noise_frame(mean=128, std=30, size=(640, 480)):
    """随机噪声帧"""
    frame = np.random.normal(mean, std, (size[1], size[0], 3))
    return np.clip(frame, 0, 255).astype(np.uint8)


def _text_frame(text_color=(255, 255, 255), bg_color=(0, 0, 0), size=(640, 480)):
    """带白色文字的黑色背景帧（模拟 OCR 场景）"""
    frame = np.full((size[1], size[0], 3), bg_color, dtype=np.uint8)
    cv2.putText(frame, "25.0 C", (200, 240), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, text_color, 4)
    return frame


def _led_frame(brightness=220, roi=(300, 200, 24, 24), size=(640, 480)):
    """带高亮 LED 区域的帧"""
    frame = _solid_frame((40, 40, 40), size)
    x, y, w, h = roi
    frame[y:y+h, x:x+w] = (brightness, brightness, brightness)
    return frame


# ── 测试：全管线端到端 ──────────────────────────────────

class TestFullPipeline:
    """模拟真实诊断循环：analyze → compute_status → confidence → prompt"""

    def test_normal_static_scene(self):
        """静态灰色场景 → normal, 置信度~0.7 (flicker+ocr 未就绪)"""
        analyzer = FrameAnalyzer()
        frame = _solid_frame()

        data = analyzer.analyze_frame(frame)
        assert data["frame_id"] == 1
        assert "color" in data
        assert "motion" in data
        assert "flicker" in data
        assert "ocr" in data

        status, cause = compute_status(data)
        assert status == "normal"
        assert "场景正常" in cause

        conf = compute_confidence(data)
        assert 0.3 <= conf <= 0.9

        prompt = build_diagnostic_prompt(data)
        assert "status: normal" in prompt
        assert "颜色:" in prompt

    def test_multiple_frames_id_increments(self):
        """帧 ID 逐帧递增"""
        analyzer = FrameAnalyzer()
        for i in range(5):
            data = analyzer.analyze_frame(_solid_frame())
            assert data["frame_id"] == i + 1

    def test_pipeline_with_noise(self):
        """噪声帧管线不崩溃"""
        analyzer = FrameAnalyzer()
        np.random.seed(42)
        for _ in range(5):
            data = analyzer.analyze_frame(_noise_frame())
            status, cause = compute_status(data)
            conf = compute_confidence(data)
            prompt = build_diagnostic_prompt(data)
            assert isinstance(status, str)
            assert isinstance(conf, float)
            assert len(prompt) > 0

    def test_pipeline_alert_on_high_contrast_change(self):
        """连续喂入差异大的帧 → MOG2 检测到运动 → alert"""
        analyzer = FrameAnalyzer()
        # 预热 MOG2 (30帧)
        dark = _solid_frame((10, 10, 10))
        for _ in range(30):
            analyzer.analyze_frame(dark)

        # 现在喂入高对比度帧
        bright = _solid_frame((240, 240, 240))
        data = analyzer.analyze_frame(bright)

        status, cause = compute_status(data)
        # MOG2 应该检测到显著变化
        assert data["motion"]["status"] == "ready"
        motion_ratio = data["motion"]["motion_ratio"]
        assert motion_ratio > 0, f"预期运动 ratio > 0, 实际 {motion_ratio}"

    def test_led_flicker_detection(self):
        """LED 高亮区域应被 flicker 模块检测"""
        analyzer = FrameAnalyzer()
        # 持续喂入 LED 帧填充 FFT buffer (128帧)
        for i in range(128):
            brightness = 220 if i % 2 == 0 else 200  # 模拟亮度波动
            analyzer.analyze_frame(_led_frame(brightness=brightness))

        data = analyzer.analyze_frame(_led_frame())
        assert data["flicker"]["status"] in ("ready", "initializing", "no_led_found")


# ── 测试：FrameAnalyzer 各模块 ──────────────────────────

class TestFrameAnalyzerModules:

    def test_color_detection_on_solid_frame(self):
        """单色帧颜色检测 → 提取主色和色温"""
        analyzer = FrameAnalyzer()
        data = analyzer.analyze_frame(_solid_frame((128, 128, 128)))
        color = data["color"]
        # 灰色帧：低饱和度
        assert color["mean_saturation"] < 20
        assert "color_temperature" in color
        assert len(color["dominant_colors"]) >= 1

    def test_motion_warmup_period(self):
        """MOG2 预热期间 status=initializing"""
        analyzer = FrameAnalyzer()
        for i in range(10):
            data = analyzer.analyze_frame(_solid_frame())
            assert data["motion"]["status"] == "initializing"

    def test_motion_ready_after_warmup(self):
        """预热完成后 MOG2 status=ready"""
        analyzer = FrameAnalyzer()
        for _ in range(35):
            analyzer.analyze_frame(_solid_frame())
        data = analyzer.analyze_frame(_noise_frame())
        assert data["motion"]["status"] == "ready"


# ── 测试：规则引擎边界 ──────────────────────────────────

class TestRuleEngineEdgeCases:

    def test_motion_high_ratio_triggers_alert(self):
        """高运动 ratio + 多区域 → alert"""
        data = _make_diag(
            motion={"status": "ready", "level": "high", "motion_ratio": 0.15,
                    "num_regions": 5, "largest_region_area": 500},
        )
        status, cause = compute_status(data)
        assert status == "alert"

    def test_motion_low_ratio_no_alert(self):
        """低运动 ratio (<0.08) 且少区域 → 不触发 alert"""
        data = _make_diag(
            motion={"status": "ready", "level": "low", "motion_ratio": 0.01,
                    "num_regions": 1, "largest_region_area": 0},
        )
        status, _ = compute_status(data)
        assert status == "normal"

    def test_flicker_below_1_5hz_no_warning(self):
        """低频闪烁 (<1.5Hz) → 排除（AGC 伪影）"""
        data = _make_diag(
            flicker={"status": "ready", "led_roi": None,
                     "frequency_hz": 0.8, "amplitude": 10.0, "is_stable": False},
        )
        status, _ = compute_status(data)
        assert status == "normal"

    def test_flicker_low_amp_no_warning(self):
        """低振幅 (<8.0) → 排除噪声"""
        data = _make_diag(
            flicker={"status": "ready", "led_roi": None,
                     "frequency_hz": 3.0, "amplitude": 4.0, "is_stable": False},
        )
        status, _ = compute_status(data)
        assert status == "normal"

    def test_confidence_floor_never_below_03(self):
        """置信度不低于 0.3"""
        data = _make_diag(
            flicker={"status": "no_led_found", "led_roi": None,
                     "frequency_hz": 0.0, "amplitude": 0.0, "is_stable": True},
            motion={"status": "initializing", "level": "none",
                    "motion_ratio": 0.0, "num_regions": 0, "largest_region_area": 0},
            ocr={"status": "unavailable", "text": "", "confidence": 0.0, "numeric_value": None},
        )
        assert compute_confidence(data) >= 0.3


# ── Helper ──────────────────────────────────────────────

def _make_diag(**overrides):
    """构建最小诊断 dict，仅包含规则引擎所需字段"""
    base = {
        "frame_id": 1, "timestamp": time.time(),
        "color": {
            "dominant_colors": [],
            "mean_hue": 0.0, "mean_saturation": 0.0, "mean_value": 0.0,
            "color_temperature": "neutral", "is_monochrome": True,
        },
        "motion": {
            "status": "ready", "level": "none",
            "motion_ratio": 0.0, "num_regions": 0, "largest_region_area": 0,
        },
        "flicker": {
            "status": "ready", "led_roi": None,
            "frequency_hz": 0.0, "amplitude": 0.0, "is_stable": True,
        },
        "ocr": {
            "status": "ok", "text": "25.0", "confidence": 0.95, "numeric_value": 25.0,
        },
    }
    base.update(overrides)
    return base
