"""JSON 容错提取单元测试 — 不依赖 Ollama 服务"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import OllamaInference

extract = OllamaInference._extract_json


def test_l1_direct_json():
    """L1: 标准 JSON 直接解析"""
    assert extract('{"status":"normal"}') == {"status": "normal"}


def test_l2_markdown_fence():
    """L2: 去除 ```json 围栏"""
    result = extract('```json\n{"status":"normal"}\n```')
    assert result == {"status": "normal"}


def test_l2_markdown_fence_no_lang():
    """L2: 去除 ``` 无语言标记围栏"""
    result = extract('```\n{"status":"alert"}\n```')
    assert result == {"status": "alert"}


def test_l3_braces_extraction():
    """L3: 从末尾附加文本中提取 JSON 对象"""
    result = extract('{"status":"warning"}\n根据分析，当前场景有轻微运动。')
    assert result == {"status": "warning"}


def test_l4_trailing_comma():
    """L4: 修复尾部逗号"""
    result = extract('{"status":"normal","details":{"color":"正常",}}')
    assert result == {"status": "normal", "details": {"color": "正常"}}


def test_l4_trailing_comma_array():
    """L4: 修复数组尾部逗号"""
    result = extract('{"values":[1,2,3,]}')
    assert result == {"values": [1, 2, 3]}


def test_nested_json():
    """嵌套 JSON 对象完整提取"""
    result = extract('{"status":"alert","confidence":0.85,"details":{"motion":"高","led":"闪烁"}}')
    assert result["status"] == "alert"
    assert result["confidence"] == 0.85
    assert result["details"]["motion"] == "高"


def test_raises_on_no_json():
    """无 JSON 时抛出 ValueError"""
    import pytest
    with pytest.raises(ValueError, match="未找到JSON"):
        extract("纯文本回复，没有 JSON")


def test_realistic_output():
    """模拟真实场景 LLM 输出（末尾带解释文字）"""
    raw = (
        '{"status":"warning","cause":"检测到轻微运动(区域数=2)","confidence":0.8,'
        '"details":{"color":"色温正常","motion":"低等级运动","flicker":"稳定","ocr":"无文字"}}'
        '\n\n以上是根据视觉数据生成的诊断结果。'
    )
    result = extract(raw)
    assert result["status"] == "warning"
    assert result["confidence"] == 0.8
