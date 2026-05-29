"""
阶段3：推理引擎原型 v4
─────────────────────
管线: Camera → FrameAnalyzer → Python规则判定 → OllamaInference(仅格式化JSON) → 输出

架构决策: Python处理确定性逻辑(规则判定+置信度), LLM仅负责JSON格式化+语义摘要
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from camera import Camera
from preprocess import FrameAnalyzer
from inference import OllamaInference


# ═══════════════════════════════════════════════════════════════
# System Prompt v4 — LLM只做JSON格式化，不做规则判断
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是JSON格式化助手。输入包含status/cause/confidence和视觉数据，输出一条JSON。

规则: 直接使用输入的status、cause、confidence值，不要修改。details中每项写简短中文摘要(≤15字)。

输出格式:
{"status":"...","cause":"...","confidence":0.0,"details":{"color":"...","motion":"...","flicker":"...","ocr":"..."}}"""


# ═══════════════════════════════════════════════════════════════
# Python确定性逻辑 — 规则判定 + 置信度计算
# ═══════════════════════════════════════════════════════════════

def compute_status(data: dict) -> tuple:
    """
    Python确定性规则判定。返回 (status, cause)。
    所有条件由Python精确计算，消除LLM误判。
    """
    motion = data["motion"]
    flicker = data["flicker"]

    # 规则1: 运动告警（最高优先级）
    if motion["status"] == "ready":
        level = motion["level"]
        regions = motion["num_regions"]
        ratio = motion["motion_ratio"]
        if (level in ("medium", "high")) and ratio >= 0.08:
            return ("alert", f"检测到显著运动(ratio={ratio:.4f}, 区域数={regions})")
        if level == "low" and regions >= 2:
            return ("warning", f"检测到轻微运动(区域数={regions})")

    # 规则2: LED闪烁异常（中等优先级）
    # 低频率(0.5-1.5Hz)多为摄像头AGC呼吸伪影，提高频率下限和振幅阈值排除噪声
    if flicker["status"] == "ready":
        freq = flicker["frequency_hz"]
        is_stable = flicker["is_stable"]
        amp = flicker["amplitude"]
        if not is_stable and 1.5 <= freq <= 5.0 and amp >= 8.0:
            return ("warning", f"LED闪烁异常(频率={freq:.2f}Hz)")

    # 规则3: 正常（兜底）
    return ("normal", "场景正常")


def compute_confidence(data: dict) -> float:
    """基于模块就绪状态计算置信度。基准0.9，逐项扣减，最低0.3。"""
    score = 0.9
    if data["flicker"]["status"] != "ready":
        score -= 0.2
    if data["ocr"]["status"] != "ok":
        score -= 0.1
    if data["motion"]["status"] != "ready":
        score -= 0.2
    return max(0.3, round(score, 2))


# ═══════════════════════════════════════════════════════════════
# build_diagnostic_prompt — 注入Python预计算结果
# ═══════════════════════════════════════════════════════════════

def build_diagnostic_prompt(data: dict) -> str:
    """
    构建prompt: Python已计算status/cause/confidence
    → LLM只需格式化JSON + 写details摘要
    """
    color = data["color"]
    motion = data["motion"]
    flicker = data["flicker"]
    ocr = data["ocr"]

    # Python预计算
    status, cause = compute_status(data)
    conf = compute_confidence(data)

    lines = [
        f"status: {status}",
        f"cause: {cause}",
        f"confidence: {conf}",
        "",
        "视觉数据(用于填充details摘要):",
    ]

    # ── 颜色 ──
    parts = [f"色温={color['color_temperature']}"]
    if color["dominant_colors"]:
        top3 = color["dominant_colors"][:3]
        parts.append("主色: " + ", ".join(
            f"{c['label']}{int(c['ratio']*100)}%" for c in top3
        ))
    lines.append("颜色: " + "; ".join(parts))

    # ── 运动 ──
    if motion["status"] == "initializing":
        lines.append("运动: 初始化中")
    else:
        parts = [f"等级={motion['level']}", f"ratio={motion['motion_ratio']:.4f}"]
        if motion["num_regions"] > 0:
            parts.append(f"区域数={motion['num_regions']}")
        else:
            parts.append("无运动区域")
        lines.append("运动: " + "; ".join(parts))

    # ── LED闪烁 ──
    if flicker["status"] == "initializing":
        lines.append("LED闪烁: 初始化中")
    elif flicker["status"] == "no_led_found":
        lines.append("LED闪烁: 未检测到LED")
    else:
        parts = [
            f"频率={flicker['frequency_hz']:.2f}Hz",
            "稳定" if flicker["is_stable"] else "不稳定",
        ]
        lines.append("LED闪烁: " + "; ".join(parts))

    # ── OCR ──
    if ocr["status"] == "disabled":
        lines.append("OCR: 已禁用")
    elif ocr["status"] == "unavailable":
        lines.append("OCR: 不可用")
    elif ocr["status"] == "no_text_found":
        lines.append("OCR: 无文字")
    elif ocr["status"] == "ok":
        parts = [f'文本="{ocr["text"]}"']
        if ocr["numeric_value"] is not None:
            parts.append(f"数值={ocr['numeric_value']}")
        lines.append("OCR: " + "; ".join(parts))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

def main():
    config_path = SRC_DIR / "config.yaml"
    config = {}
    if config_path.exists():
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    stage3_cfg = config.get("stage3", {})
    inference_cfg = config.get("inference", {})
    inference_interval = stage3_cfg.get("inference_interval", 30)
    warmup_frames = stage3_cfg.get("warmup_frames", 158)
    temperature = stage3_cfg.get("temperature", 0.2)

    print("=" * 60)
    print("  Vantage 阶段3：推理引擎原型 v4")
    print(f"  架构: Python规则判定 + LLM格式化JSON")
    print(f"  推理间隔: 每 {inference_interval} 帧")
    print(f"  预热帧数: {warmup_frames} 帧")
    print("=" * 60)

    inference = OllamaInference(
        model=inference_cfg.get("model", "vantage"),
        base_url=os.environ.get("OLLAMA_URL") or inference_cfg.get("base_url", "http://localhost:11434"),
        cold_timeout=inference_cfg.get("cold_timeout", 300),
        hot_timeout=inference_cfg.get("hot_timeout", 120),
    )

    if not inference.ping():
        print("\n[FATAL] Ollama 服务未运行。请先执行: ollama serve")
        return

    print("\n[1/3] 预热LLM模型...")
    t0 = time.time()
    inference.warmup()
    print(f"      耗时 {time.time() - t0:.1f}s")

    print(f"[2/3] 打开摄像头 + 预热视觉管线 ({warmup_frames}帧)...")
    with Camera() as cam:
        analyzer = FrameAnalyzer(config.get("preprocess", {}))

        for i in range(warmup_frames):
            frame = cam.capture()
            analyzer.analyze_frame(frame)
            if (i + 1) % 30 == 0:
                print(f"      进度: {i + 1}/{warmup_frames}")

        print(f"[3/3] 预热完成。首次推理将在第 "
              f"{((warmup_frames // inference_interval) + 1) * inference_interval} 帧触发\n")

        total = 0
        ok = 0
        fail = 0
        latencies = []

        print("按 Ctrl+C 停止推理循环\n")
        print("-" * 60)

        try:
            while True:
                frame = cam.capture()
                data = analyzer.analyze_frame(frame)
                fid = data["frame_id"]

                if fid % inference_interval != 0:
                    continue

                py_status, py_cause = compute_status(data)
                py_conf = compute_confidence(data)

                prompt_text = build_diagnostic_prompt(data)
                t0 = time.time()

                try:
                    result = inference.generate(
                        prompt=prompt_text,
                        system=SYSTEM_PROMPT,
                        temperature=temperature,
                        json_mode=True,
                    )
                    elapsed = time.time() - t0
                    latencies.append(elapsed)
                    total += 1

                except ValueError as e:
                    elapsed = time.time() - t0
                    print(f"\n[帧 {fid}] JSON提取失败: {str(e)[:120]}")
                    fail += 1
                    total += 1
                    continue

                required = {"status", "cause", "confidence", "details"}
                missing = required - set(result.keys())
                if missing:
                    print(f"\n[帧 {fid}] 输出缺少字段: {missing}")
                    fail += 1
                else:
                    ok += 1

                llm_status = result.get("status", "?")
                match_mark = "✓" if llm_status == py_status else f"✗(应为{py_status})"

                icon = {"normal": "✅", "warning": "⚠️", "alert": "🚨"}.get(llm_status, "❓")
                print(f"\n{'─' * 50}")
                print(f"  帧 #{fid} | 耗时 {elapsed:.1f}s | {icon} {llm_status} {match_mark}")
                print(f"  原因: {result.get('cause', 'N/A')}")
                print(f"  置信度: {result.get('confidence', 0):.2f} (Python: {py_conf})")
                details = result.get("details", {})
                if isinstance(details, dict):
                    for k, v in details.items():
                        print(f"    [{k}] {v}")
                print(f"{'─' * 50}")

        except KeyboardInterrupt:
            print("\n\n用户中断 (Ctrl+C)。")

        print("\n" + "=" * 60)
        print("  阶段3运行统计")
        print("=" * 60)
        print(f"  总推理次数:       {total}")
        if total > 0:
            print(f"  JSON合法+完整:    {ok}  ({ok/total*100:.0f}%)")
            print(f"  JSON失败/缺字段:  {fail}  ({fail/total*100:.0f}%)")
            if latencies:
                avg_lat = sum(latencies) / len(latencies)
                print(f"  推理延迟:         平均 {avg_lat:.1f}s / 最小 {min(latencies):.1f}s / 最大 {max(latencies):.1f}s")
            print(f"\n  {'✅' if ok/total >= 0.9 else '❌'} JSON合规率 {ok/total*100:.0f}% {'≥' if ok/total >= 0.9 else '<'} 90%")
            print(f"  {'✅' if latencies and sum(latencies)/len(latencies) <= 6.0 else '❌'} 平均延迟 {avg_lat:.1f}s {'≤' if avg_lat <= 6.0 else '>'} 6s")
        else:
            print("  (无推理记录)")
        print("=" * 60)


if __name__ == "__main__":
    main()
