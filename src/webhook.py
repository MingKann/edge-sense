"""
Webhook 告警通知模块
───────────────────────
当诊断结果匹配配置的状态等级时，向指定 URL 发送 HTTP POST 请求。

特性:
  - 多目标支持：可配置多个 webhook URL
  - 可配置触发事件：alert / warning / 两者
  - 指数退避重试：最多 2 次重试
  - 错误隔离：webhook 失败不影响主管线
  - 非阻塞：在独立线程中执行

配置示例 (config.yaml):
  webhook:
    enabled: false
    on_status: ["alert"]
    timeout: 5
    retries: 2
    urls:
      - url: "https://hooks.example.com/edge-sense"
        headers:
          X-Custom-Header: "value"
"""

import json
import logging
import time
import threading
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("webhook")


class WebhookConfig:
    """Webhook 配置模型"""

    def __init__(self, cfg: dict):
        self.enabled: bool = cfg.get("enabled", False)
        self.on_status: List[str] = cfg.get("on_status", ["alert"])
        self.timeout: int = cfg.get("timeout", 5)
        self.retries: int = cfg.get("retries", 2)
        self.urls: List[Dict] = cfg.get("urls", [])

    @property
    def active(self) -> bool:
        """是否启用且有至少一个目标 URL"""
        return self.enabled and len(self.urls) > 0


def build_payload(diagnosis: dict) -> dict:
    """
    从诊断数据构建 webhook 载荷

    返回结构化的 JSON，方便下游系统消费（如 Discord、Slack、自建服务）。
    """
    return {
        "event": "diagnosis",
        "timestamp": diagnosis.get("timestamp", time.time()),
        "frame_id": diagnosis.get("frame_id"),
        "status": diagnosis.get("status"),
        "cause": diagnosis.get("cause"),
        "confidence": diagnosis.get("confidence"),
        "inference_time_s": diagnosis.get("inference_time_s"),
        "details": {
            "color": {
                "temperature": diagnosis.get("color", {}).get("color_temperature"),
                "dominant": diagnosis.get("color", {}).get("dominant_colors", [])[:2],
            },
            "motion": {
                "level": diagnosis.get("motion", {}).get("level"),
                "ratio": diagnosis.get("motion", {}).get("motion_ratio"),
                "regions": diagnosis.get("motion", {}).get("num_regions"),
            },
            "flicker": {
                "frequency_hz": diagnosis.get("flicker", {}).get("frequency_hz"),
                "stable": diagnosis.get("flicker", {}).get("is_stable"),
            },
            "ocr": {
                "text": diagnosis.get("ocr", {}).get("text"),
                "confidence": diagnosis.get("ocr", {}).get("confidence"),
            },
        },
        "source": {
            "name": "edge-sense",
            "version": "0.4.0",
        },
    }


def send_webhook(
    url_config: dict,
    payload: dict,
    timeout: int = 5,
    retries: int = 2,
) -> bool:
    """
    向单个 webhook URL 发送通知

    参数:
        url_config: {"url": "...", "headers": {...}}
        payload:    要发送的 JSON 体
        timeout:    请求超时秒数
        retries:    失败重试次数（指数退避）

    返回:
        True  = 发送成功（HTTP 2xx）
        False = 发送失败（所有重试用尽）
    """
    url = url_config.get("url", "")
    headers = dict(url_config.get("headers", {}))
    headers.setdefault("Content-Type", "application/json")
    headers.setdefault("User-Agent", "Edge-Sense/0.4.0 Webhook")

    for attempt in range(1 + retries):
        try:
            resp = requests.post(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code < 300:
                logger.info(f"Webhook OK ({resp.status_code}): {url}")
                return True
            else:
                logger.warning(
                    f"Webhook HTTP {resp.status_code} (attempt {attempt + 1}): {url}"
                )
        except requests.RequestException as e:
            logger.warning(f"Webhook error (attempt {attempt + 1}): {e}")

        if attempt < retries:
            sleep_time = 2 ** attempt  # 指数退避: 1s, 2s
            logger.info(f"Webhook 将在 {sleep_time}s 后重试...")
            time.sleep(sleep_time)

    logger.error(f"Webhook 失败（已用尽 {retries} 次重试）: {url}")
    return False


def dispatch_webhooks(
    config: WebhookConfig,
    diagnosis: dict,
) -> None:
    """
    异步分发 webhook 通知到所有已配置的目标

    在独立线程中执行，不阻塞调用方。
    """
    if not config.active:
        return

    status = diagnosis.get("status", "")
    if status not in config.on_status:
        return

    payload = build_payload(diagnosis)

    threads = []
    for url_cfg in config.urls:
        t = threading.Thread(
            target=send_webhook,
            args=(url_cfg, payload, config.timeout, config.retries),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # 不 join——让后台线程自行完成，不阻塞主循环
