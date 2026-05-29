"""Webhook 模块单元测试 — 不依赖外部 HTTP 服务"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from webhook import WebhookConfig, build_payload, send_webhook, dispatch_webhooks


# ── WebhookConfig ────────────────────────────────────

def test_config_disabled_by_default():
    """未配置时默认禁用"""
    cfg = WebhookConfig({})
    assert cfg.active is False


def test_config_active_when_enabled():
    """启用 + 配置 URL → 激活"""
    cfg = WebhookConfig({"enabled": True, "urls": [{"url": "http://example.com"}]})
    assert cfg.active is True


def test_config_on_status_default():
    """默认仅 alert 级别触发"""
    cfg = WebhookConfig({"enabled": True, "urls": [{"url": "http://example.com"}]})
    assert cfg.on_status == ["alert"]


def test_config_on_status_custom():
    """自定义触发级别"""
    cfg = WebhookConfig({
        "enabled": True,
        "on_status": ["alert", "warning"],
        "urls": [{"url": "http://example.com"}],
    })
    assert cfg.on_status == ["alert", "warning"]


# ── build_payload ───────────────────────────────────

def test_build_payload_structure():
    """载荷应包含 event/timestamp/status/cause 等顶层字段"""
    diag = {
        "frame_id": 42, "timestamp": 1000.0,
        "status": "alert", "cause": "检测到显著运动",
        "confidence": 0.7, "inference_time_s": 5.1,
        "color": {"color_temperature": "cool", "dominant_colors": []},
        "motion": {"level": "high", "motion_ratio": 0.15, "num_regions": 3},
        "flicker": {"frequency_hz": 0.0, "is_stable": True},
        "ocr": {"text": "", "confidence": 0.0},
    }
    payload = build_payload(diag)

    assert payload["event"] == "diagnosis"
    assert payload["frame_id"] == 42
    assert payload["status"] == "alert"
    assert payload["cause"] == "检测到显著运动"
    assert payload["source"]["name"] == "edge-sense"


def test_build_payload_color_dominant_only_top2():
    """主色只取前 2 个（控制载荷大小）"""
    diag = {
        "frame_id": 1, "timestamp": 0.0,
        "status": "normal", "cause": "", "confidence": 0.9, "inference_time_s": 0,
        "color": {
            "color_temperature": "neutral",
            "dominant_colors": [
                {"hex": "#A", "label": "红", "ratio": 0.5},
                {"hex": "#B", "label": "绿", "ratio": 0.3},
                {"hex": "#C", "label": "蓝", "ratio": 0.2},
            ],
        },
        "motion": {}, "flicker": {}, "ocr": {},
    }
    payload = build_payload(diag)
    assert len(payload["details"]["color"]["dominant"]) == 2


# ── send_webhook ─────────────────────────────────────

def test_send_webhook_http_error(caplog):
    """HTTP 非 2xx 返回 False"""
    # 使用 httpbin 的 404 端点
    result = send_webhook(
        {"url": "https://httpbin.org/status/404"},
        {"test": True}, timeout=5, retries=0,
    )
    assert result is False


def test_send_webhook_connection_error():
    """连接失败返回 False（不崩溃）"""
    result = send_webhook(
        {"url": "http://127.0.0.1:1"},  # 端口 1 必然拒绝连接
        {"test": True}, timeout=1, retries=0,
    )
    assert result is False


# ── dispatch_webhooks ──────────────────────────────

def test_dispatch_skips_when_disabled():
    """禁用时不触发任何请求（不会崩溃）"""
    cfg = WebhookConfig({"enabled": False})
    dispatch_webhooks(cfg, {"status": "alert"})
    # 无异常即通过


def test_dispatch_skips_non_matching_status():
    """状态不在 on_status 列表中时跳过"""
    cfg = WebhookConfig({
        "enabled": True,
        "on_status": ["alert"],
        "urls": [{"url": "http://127.0.0.1:1"}],
    })
    dispatch_webhooks(cfg, {"status": "normal"})
    # 无异常即通过


def test_dispatch_does_not_block():
    """dispatch 应立刻返回（后台线程执行）"""
    import time
    cfg = WebhookConfig({
        "enabled": True,
        "on_status": ["alert"],
        "urls": [{"url": "http://127.0.0.1:1"}],
    })
    t0 = time.time()
    dispatch_webhooks(cfg, {"status": "alert"})
    elapsed = time.time() - t0
    assert elapsed < 1.0  # 不应等待网络超时
