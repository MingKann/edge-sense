"""
性能基准测试 — 测量关键代码路径的执行时间

运行方式:
    pytest tests/test_benchmark.py -v --benchmark-only
    pytest tests/test_benchmark.py -v --benchmark-histogram   # 生成分布图
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage3_prototype import compute_status, compute_confidence, build_diagnostic_prompt
from inference import OllamaInference

extract = OllamaInference._extract_json


# ── 测试数据工厂 ──────────────────────────────────────

def _data(**overrides):
    """生成诊断数据的基准测试样本"""
    base = {
        "frame_id": 1000, "timestamp": 0.0,
        "color": {
            "dominant_colors": [
                {"hex": "#3FB950", "rgb": [63, 185, 80], "ratio": 0.45, "label": "绿"},
                {"hex": "#58A6FF", "rgb": [88, 166, 255], "ratio": 0.30, "label": "蓝"},
            ],
            "mean_hue": 120.0, "mean_saturation": 45.0, "mean_value": 180.0,
            "color_temperature": "cool", "is_monochrome": False,
        },
        "motion": {
            "status": "ready", "level": "none",
            "motion_ratio": 0.0000, "num_regions": 0, "largest_region_area": 0,
        },
        "flicker": {
            "status": "ready", "led_roi": (100, 200, 24, 24),
            "frequency_hz": 0.0, "amplitude": 0.5, "is_stable": True,
        },
        "ocr": {
            "status": "ok", "text": "25.0°C", "confidence": 0.92, "numeric_value": 25.0,
        },
    }
    base.update(overrides)
    return base


# ── 基准测试：compute_status ──────────────────────────

def test_bench_status_normal(benchmark):
    """normal 场景：无运动 + 无闪烁"""
    data = _data()
    benchmark(compute_status, data)

def test_bench_status_alert(benchmark):
    """alert 场景：高强度运动"""
    data = _data(motion_level="high", motion_ratio=0.15, num_regions=5)
    benchmark(compute_status, data)

def test_bench_status_warning_motion(benchmark):
    """warning 场景：轻微运动"""
    data = _data(motion_level="low", motion_ratio=0.01, num_regions=2)
    benchmark(compute_status, data)

def test_bench_status_led_flicker(benchmark):
    """LED 闪烁告警"""
    data = _data(flicker_stable=False, frequency_hz=2.5, flicker_amp=10.0)
    # 注意: 为 flicker 传参需直接改 data
    data["flicker"]["is_stable"] = False
    data["flicker"]["frequency_hz"] = 2.5
    data["flicker"]["amplitude"] = 10.0
    benchmark(compute_status, data)


# ── 基准测试：compute_confidence ─────────────────────

def test_bench_confidence_all_ready(benchmark):
    """所有模块就绪"""
    benchmark(compute_confidence, _data())

def test_bench_confidence_partial(benchmark):
    """部分模块未就绪"""
    data = _data()
    data["flicker"]["status"] = "no_led_found"
    data["ocr"]["status"] = "no_text_found"
    benchmark(compute_confidence, data)


# ── 基准测试：build_diagnostic_prompt ─────────────────

def test_bench_prompt_normal(benchmark):
    """正常场景 prompt 构建（含颜色主色 + 数值）"""
    benchmark(build_diagnostic_prompt, _data())

def test_bench_prompt_initializing(benchmark):
    """预热中场景（各模块尚未就绪）"""
    data = _data(
        motion_status="initializing",
        flicker_status="initializing",
        ocr_status="no_text_found",
    )
    benchmark(build_diagnostic_prompt, data)


# ── 基准测试：_extract_json ──────────────────────────

def test_bench_json_l1_direct(benchmark):
    """L1: 标准 JSON 直接解析"""
    benchmark(extract, '{"status":"normal","confidence":0.9,"cause":"场景正常"}')

def test_bench_json_l2_markdown(benchmark):
    """L2: Markdown 围栏"""
    benchmark(extract, '```json\n{"status":"alert","confidence":0.7}\n```')

def test_bench_json_l3_trailing_text(benchmark):
    """L3: 末尾附加文本"""
    benchmark(extract, '{"status":"warning","confidence":0.8}\n根据视觉数据分析结果，当前场景有轻微运动。')

def test_bench_json_l4_trailing_comma(benchmark):
    """L4: 尾部逗号修复"""
    benchmark(extract, '{"status":"alert","cause":"检测到运动","details":{"motion":"high",}}')


# ── 基准测试：完整管线模拟 ─────────────────────────

def test_bench_full_pipeline_normal(benchmark):
    """模拟一次完整诊断循环（compute_status + confidence + prompt）"""
    data = _data()

    def run():
        s = compute_status(data)
        c = compute_confidence(data)
        p = build_diagnostic_prompt(data)
        return s, c, p

    benchmark(run)
