"""
Vantage Web 面板后端 (阶段4 v7)
FastAPI + MJPEG 视频流 + WebSocket 诊断推送

v7: signal.signal(SIGINT) C级处理器兜底，无视 uvicorn 关闭等待（Ctrl+C <1秒退出）
"""

import asyncio
import os
import signal
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from camera import Camera
from preprocess import FrameAnalyzer
from inference import OllamaInference, InferenceInterrupted
from stage3_prototype import (
    SYSTEM_PROMPT,
    compute_status,
    compute_confidence,
    build_diagnostic_prompt,
)
from webhook import WebhookConfig, dispatch_webhooks
from history import get_store

# ── 加载配置 ─────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_webhook_cfg = WebhookConfig({})
if CONFIG_PATH.exists():
    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _full_cfg = yaml.safe_load(f) or {}
        _webhook_cfg = WebhookConfig(_full_cfg.get("webhook", {}))
    except Exception as e:
        print(f"[server] 配置加载失败: {e}")

# ── 全局状态 ─────────────────────────────────────────────

latest_frame_jpeg: Optional[bytes] = None
latest_diagnosis: Optional[dict] = None
frame_lock = threading.Lock()
diag_lock = threading.Lock()
shutdown_event = threading.Event()
ready = threading.Event()

INFERENCE_INTERVAL = 30
WARMUP_FRAMES = 158
JPEG_QUALITY = 70

# 自适应推理频率：运动越强，推理越频繁
ADAPTIVE_INTERVALS = {"none": 60, "low": 30, "medium": 15, "high": 15}
INTERVAL_HYSTERESIS = 3  # 连续N帧同等级后才切换，防止振荡

# 从配置覆盖自适应参数
if CONFIG_PATH.exists():
    try:
        _stage3 = _full_cfg.get("stage3", {})
        _adaptive = _stage3.get("adaptive", {})
        if _adaptive:
            for k in ("none", "low", "medium", "high"):
                if k in _adaptive:
                    ADAPTIVE_INTERVALS[k] = _adaptive[k]
            INTERVAL_HYSTERESIS = _adaptive.get("hysteresis", INTERVAL_HYSTERESIS)
    except Exception:
        pass

# ── 后台线程 ────────────────────────────────────────────

def camera_loop():
    global latest_frame_jpeg, latest_diagnosis
    print("[camera-thread] 启动...")

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    print(f"[camera-thread] Ollama URL: {ollama_url}")
    inference = OllamaInference(model="vantage", base_url=ollama_url,
                                cold_timeout=300, hot_timeout=120)

    if not inference.ping():
        print("[camera-thread] FATAL: Ollama 服务未运行")
        ready.set()
        return

    print("[camera-thread] 预热LLM模型...")
    inference.warmup()

    print(f"[camera-thread] 预热视觉管线 ({WARMUP_FRAMES}帧)...")
    cam = Camera()
    cam.open()
    analyzer = FrameAnalyzer()

    try:
        for i in range(WARMUP_FRAMES):
            if shutdown_event.is_set():
                print("[camera-thread] 预热阶段收到退出信号")
                return
            frame = cam.capture()
            analyzer.analyze_frame(frame)

        print("[camera-thread] 预热完成，进入主循环")
        ready.set()

        # 自适应推理：运动越强频率越高，静态时降低以节省资源
        interval = INFERENCE_INTERVAL
        last_motion_level = "none"
        level_streak = 0
        next_inference_at = WARMUP_FRAMES + interval

        while not shutdown_event.is_set():
            frame = cam.capture()
            data = analyzer.analyze_frame(frame)
            fid = data["frame_id"]

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with frame_lock:
                latest_frame_jpeg = jpeg.tobytes()

            # ── 自适应频率调整 ──────────────────────
            motion_level = data["motion"]["level"]
            if motion_level == last_motion_level:
                level_streak += 1
            else:
                last_motion_level = motion_level
                level_streak = 1

            if level_streak >= INTERVAL_HYSTERESIS:
                new_interval = ADAPTIVE_INTERVALS.get(motion_level, 30)
                if new_interval != interval:
                    interval = new_interval
                    print(f"[camera-thread] 推理间隔 → {interval}帧 (运动={motion_level})")

            # 推理帧（退出前跳过推理，避免在HTTP请求中卡住）
            if fid >= next_inference_at and not shutdown_event.is_set():
                next_inference_at = fid + interval
                py_status, py_cause = compute_status(data)
                py_conf = compute_confidence(data)
                prompt_text = build_diagnostic_prompt(data)
                t0 = time.time()

                try:
                    llm_result = inference.generate(
                        prompt=prompt_text, system=SYSTEM_PROMPT,
                        temperature=0.2, json_mode=True,
                        shutdown_event=shutdown_event,   # ★ 传入退出信号
                    )
                except InferenceInterrupted:
                    print(f"[camera-thread] 推理期间收到退出信号，停止采集")
                    break                                 # ★ 立即退出主循环
                except Exception as e:
                    print(f"[camera-thread] 推理异常: {e}")
                    llm_result = {}

                diag = {
                    "frame_id": fid, "timestamp": time.time(),
                    "inference_time_s": round(time.time() - t0, 2),
                    "status": llm_result.get("status", py_status),
                    "cause": llm_result.get("cause", py_cause),
                    "confidence": llm_result.get("confidence", py_conf),
                    "details": llm_result.get("details", {}),
                    "color": data["color"], "motion": data["motion"],
                    "flicker": data["flicker"], "ocr": data["ocr"],
                }

                with diag_lock:
                    latest_diagnosis = diag

                print(f"[camera-thread] 帧#{fid} | {diag['status']} | "
                      f"{diag['inference_time_s']}s | {diag['cause'][:40]}")

                # Webhook 异步通知（alert/warning 时触发）
                dispatch_webhooks(_webhook_cfg, diag)

                # 持久化到 SQLite 历史
                try:
                    get_store().save(diag)
                except Exception as e:
                    print(f"[camera-thread] 历史写入失败: {e}")

    except Exception as e:
        print(f"[camera-thread] 异常退出: {e}")
    finally:
        cam.release()
        ready.clear()
        print("[camera-thread] 已退出")

# ── FastAPI 生命周期 ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    shutdown_event.clear()
    thread = threading.Thread(target=camera_loop, daemon=True, name="camera-thread")
    thread.start()
    print("[server] 等待后台线程预热...")
    yield
    print("[server] 等待后台线程退出...")
    shutdown_event.set()
    thread.join(timeout=2)
    if thread.is_alive():
        print("[server] 后台线程未在2秒内退出，daemon线程随进程终止")
    os._exit(0)  # 无视 uvicorn 连接关闭等待，立即退出

# ── FastAPI 应用 ────────────────────────────────────────

app = FastAPI(title="Vantage", version="0.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>web/index.html not found</h1>", status_code=404)


@app.get("/favicon.ico")
async def favicon():
    """Vantage SVG favicon — 消除浏览器 404 请求"""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#161b22"/>'
        '<text x="16" y="23" text-anchor="middle" '
        'font-family="Arial" font-weight="bold" font-size="20" fill="#58a6ff">E</text>'
        '</svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/video_feed")
async def video_feed():
    """MJPEG视频流 — 不依赖shutdown_event，uvicorn关闭连接时自然退出"""

    async def generate():
        while True:
            with frame_lock:
                jpeg = latest_frame_jpeg
            if jpeg is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_diagnosis(websocket: WebSocket):
    await websocket.accept()
    print("[ws] 客户端已连接")

    if not ready.is_set():
        await websocket.send_json({"status": "warming_up", "message": "模型预热中..."})
        ready.wait(timeout=300)
        if not ready.is_set():
            await websocket.send_json({"status": "error", "message": "预热超时"})
            await websocket.close()
            return
        await websocket.send_json({"status": "ready", "message": "预热完成"})

    last_fid = -1
    try:
        while not shutdown_event.is_set():
            with diag_lock:
                diag = latest_diagnosis
            if diag is not None and diag.get("frame_id", -1) != last_fid:
                last_fid = diag["frame_id"]
                payload = {
                    "frame_id": diag["frame_id"], "status": diag["status"],
                    "cause": diag["cause"], "confidence": diag["confidence"],
                    "inference_time_s": diag["inference_time_s"], "details": diag["details"],
                    "color": diag["color"], "motion": diag["motion"],
                    "flicker": diag["flicker"], "ocr": diag["ocr"],
                }
                await websocket.send_json(payload)
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        print("[ws] 客户端已断开")


@app.post("/shutdown")
async def shutdown():
    """立即终止服务进程"""
    print("[server] 收到 /shutdown 请求，立即退出")
    shutdown_event.set()
    os._exit(0)


# ── 历史查询 API ───────────────────────────────────────

@app.get("/api/history")
async def api_history(limit: int = 100, offset: int = 0):
    """查询诊断历史记录（按时间倒序）"""
    try:
        store = get_store()
        rows = store.query(limit=min(limit, 1000), offset=offset)
        return {"count": len(rows), "rows": rows}
    except Exception as e:
        return {"error": str(e), "rows": []}


@app.get("/api/history/stats")
async def api_history_stats():
    """诊断聚合统计"""
    try:
        return get_store().stats()
    except Exception as e:
        return {"error": str(e), "total": 0}


@app.get("/api/history/trend")
async def api_history_trend(limit: int = 50):
    """置信度时间序列（用于前端趋势图）"""
    try:
        return {
            "series": get_store().recent_confidence_series(limit=min(limit, 200))
        }
    except Exception as e:
        return {"error": str(e), "series": []}

# ── 入口 ────────────────────────────────────────────────

def _handle_sigint(signum, frame):
    """
    Ctrl+C 信号处理器（C级兜底）

    Windows 上 uvicorn 的 lifespan shutdown 可能不被 Ctrl+C 正确触发。
    本处理器在信号到达时立即终止进程，无视 uvicorn 的连接等待/关闭流程。

    os._exit(0) 不触发 Python 退出处理，直接向内核发起 exit 系统调用，
    所有 daemon 线程随进程一并终止。Ollama 服务端不受影响（模型已驻留显存）。
    """
    print("\n[server] Ctrl+C 信号，立即退出")
    shutdown_event.set()
    os._exit(0)


if __name__ == "__main__":
    # 注册信号处理器（必须在 uvicorn 接管之前）
    signal.signal(signal.SIGINT, _handle_sigint)

    print("=" * 50)
    print("  Vantage Web Panel (阶段4 v7)")
    print("  前端: http://localhost:8000")
    print("  停止: Ctrl+C 或 POST /shutdown（<1秒内退出）")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
