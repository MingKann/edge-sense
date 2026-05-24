"""阶段0冒烟测试：摄像头 + 模型加载 + 一次推理"""
import cv2
from llama_cpp import Llama
import os
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
cap.release()
model_path = os.path.join("models", "qwen2.5-3b-instruct-q4_k_m.gguf")
print(f"[TEST] Loading model from {model_path}...")
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
print("[OK] Model loaded")
print("[TEST] Running inference...")
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "用一句话回答：什么是边缘计算？"}],
    max_tokens=100,
    temperature=0.7
)
answer = response["choices"][0]["message"]["content"].strip()
print(f"[OK] LLM reply ({len(answer)} chars): {answer[:80]}...")
llm.close()
print()
print("=" * 40)
print("阶段零完成！所有依赖正常工作。")
print("=" * 40)
