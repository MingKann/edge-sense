"""阶段0冒烟测试：摄像头 + Ollama 推理"""
import cv2
import requests

print("[TEST] Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[FAIL] No camera found.")
    exit(1)
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f"[OK] Camera working: {w}x{h}")
else:
    print("[FAIL] Camera opened but failed to read frame.")
    cap.release()
    exit(1)
cap.release()

print("[TEST] Testing Ollama...")
try:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "vantage",
            "prompt": "用一句话回答：什么是边缘计算？",
            "stream": False
        },
        timeout=300
    )
    answer = resp.json()["response"].strip()
    print(f"[OK] Ollama reply ({len(answer)} chars): {answer[:80]}...")
except Exception as e:
    print(f"[FAIL] Ollama error: {e}")
    exit(1)

print()
print("=" * 40)
print("阶段零完成！所有依赖正常工作。")
print("=" * 40)
