"""规则引擎单元测试 — 不依赖硬件 (无需摄像头/Ollama)"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage3_prototype import compute_status, compute_confidence, build_diagnostic_prompt


# ── compute_status ──────────────────────────────────────

def test_status_normal_when_no_motion():
    """无运动 + 无闪烁 → normal"""
    data = make_data(motion_level="none", motion_ratio=0.0, flicker_stable=True)
    status, cause = compute_status(data)
    assert status == "normal"


def test_status_alert_high_motion():
    """medium/high 运动 + ratio≥0.08 → alert"""
    data = make_data(motion_level="medium", motion_ratio=0.09, regions=3)
    status, cause = compute_status(data)
    assert status == "alert"
    assert "显著运动" in cause


def test_status_alert_high_motion_low_ratio():
    """medium 运动但 ratio<0.08 → 不触发 alert"""
    data = make_data(motion_level="medium", motion_ratio=0.05)
    status, _ = compute_status(data)
    assert status != "alert"


def test_status_warning_low_motion():
    """low 运动 + 区域≥2 → warning"""
    data = make_data(motion_level="low", motion_ratio=0.01, regions=2)
    status, cause = compute_status(data)
    assert status == "warning"
    assert "轻微运动" in cause


def test_status_warning_led_flicker():
    """LED 不稳定 + 1.5≤freq≤5.0 + amp≥8.0 → warning"""
    data = make_data(
        motion_level="none", flicker_stable=False,
        flicker_freq=3.0, flicker_amp=10.0,
    )
    status, cause = compute_status(data)
    assert status == "warning"
    assert "LED" in cause


def test_status_led_flicker_low_amp():
    """LED 不稳定但 amp<8.0 → 不触发（排除 AGC 伪影）"""
    data = make_data(
        motion_level="none", flicker_stable=False,
        flicker_freq=3.0, flicker_amp=4.0,
    )
    status, _ = compute_status(data)
    assert status == "normal"


def test_status_motion_overrides_led():
    """运动 alert 优先级高于 LED warning"""
    data = make_data(
        motion_level="high", motion_ratio=0.15,
        flicker_stable=False, flicker_freq=3.0, flicker_amp=10.0,
    )
    status, _ = compute_status(data)
    assert status == "alert"


# ── compute_confidence ──────────────────────────────────

def test_confidence_all_ready():
    """所有模块就绪 → 0.9"""
    data = make_data()
    assert compute_confidence(data) == 0.9


def test_confidence_flicker_not_ready():
    """闪烁模块未就绪 → -0.2"""
    data = make_data(flicker_status="initializing")
    assert compute_confidence(data) == 0.7


def test_confidence_ocr_not_ok():
    """OCR 非 ok → -0.1"""
    data = make_data(ocr_status="no_text_found")
    assert compute_confidence(data) == 0.8


def test_confidence_floor():
    """多模块异常 → 最低 0.3"""
    data = make_data(
        flicker_status="no_led_found",
        motion_status="initializing",
        ocr_status="unavailable",
    )
    assert compute_confidence(data) == 0.4


# ── build_diagnostic_prompt ─────────────────────────────

def test_build_prompt_includes_status():
    """prompt 包含 Python 预计算的 status"""
    data = make_data(motion_level="high", motion_ratio=0.15)
    prompt = build_diagnostic_prompt(data)
    assert "status: alert" in prompt
    assert "cause:" in prompt
    assert "confidence:" in prompt


def test_build_prompt_includes_sensor_data():
    """prompt 包含各模块视觉数据"""
    data = make_data()
    prompt = build_diagnostic_prompt(data)
    assert "颜色:" in prompt
    assert "运动:" in prompt
    assert "LED闪烁:" in prompt
    assert "OCR:" in prompt


# ── Helper ──────────────────────────────────────────────

def make_data(
    motion_level="none", motion_ratio=0.0, regions=0,
    motion_status="ready",
    flicker_status="ready", flicker_stable=True,
    flicker_freq=0.0, flicker_amp=0.0,
    ocr_status="ok", ocr_text="25.0", ocr_numeric=25.0,
):
    return {
        "frame_id": 1,
        "timestamp": 0.0,
        "color": {
            "dominant_colors": [],
            "mean_hue": 0.0, "mean_saturation": 0.0, "mean_value": 0.0,
            "color_temperature": "neutral", "is_monochrome": True,
        },
        "motion": {
            "status": motion_status, "level": motion_level,
            "motion_ratio": motion_ratio, "num_regions": regions,
            "largest_region_area": 0,
        },
        "flicker": {
            "status": flicker_status, "led_roi": None,
            "frequency_hz": flicker_freq, "amplitude": flicker_amp,
            "is_stable": flicker_stable,
        },
        "ocr": {
            "status": ocr_status, "text": ocr_text,
            "confidence": 0.95, "numeric_value": ocr_numeric,
        },
    }
