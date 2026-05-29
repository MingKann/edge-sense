# Vantage — 边缘视觉诊断系统

## 项目概述

通过USB摄像头采集实时画面，经OpenCV预处理管线（颜色/运动/闪烁/OCR）提取特征，Python规则引擎做确定性判定，Ollama本地大模型做JSON格式化输出，最终在Web仪表盘展示诊断结果。

**架构原则**: Python处理逻辑，LLM只做格式化。不依赖LLM做规则判断或数值计算。

## 技术栈

- Python 3.x (Windows 11)
- OpenCV (摄像头采集 + 视觉预处理)
- Ollama + Qwen2.5-3B-Instruct (Q4_K_M量化, 2GB显存)
- FastAPI + uvicorn (Web服务 + MJPEG流 + WebSocket)
- Tesseract OCR (UB-Mannheim Windows安装版)
- 前端: 单文件HTML暗色仪表盘

## 目录结构

```
vantage/
├── src/
│   ├── camera.py          # 摄像头采集 (Camera类, DSHOW后端)
│   ├── preprocess.py      # 预处理管线 (FrameAnalyzer: 颜色+运动+闪烁+OCR)
│   ├── inference.py       # Ollama HTTP API封装 (流式/非流式, JSON容错提取)
│   ├── stage3_prototype.py # 阶段3: Python规则引擎 + LLM格式化
│   ├── server.py          # 阶段4: FastAPI Web面板 (MJPEG+WebSocket)
│   ├── config.yaml        # 全局配置 (摄像头/推理/预处理参数)
│   ├── stage1_prototype.py
│   ├── stage2_test.py
│   └── check.py
├── web/
│   └── index.html         # 暗色仪表盘前端
├── models/                # GGUF模型文件 (gitignored)
├── tests/
├── Modelfile.qwen         # Ollama模型定义
├── requirements.txt
└── venv/
```

## 常用命令

```bash
# 激活虚拟环境
source venv/Scripts/activate

# 启动Web面板 (主入口)
cd src && python server.py
# 访问 http://localhost:8000

# 运行阶段3原型 (终端模式, 无Web)
cd src && python stage3_prototype.py

# 检查环境
cd src && python check.py

# 模型管理
ollama list              # 查看已加载模型
ollama create vantage -f Modelfile.qwen   # 创建模型
```

## 开发约束

- venv/ 和 models/ 在 .gitignore 中
- Tesseract路径硬编码在 preprocess.py 顶部 (UB-Mannheim安装)
- 摄像头使用 CAP_DSHOW 后端 (Windows)
- Ollama模型通过 keep_alive=-1 永久驻留显存
- Ctrl+C 通过 signal.signal + os._exit(0) 立即退出
- JSON提取有四层容错回退 (inference.py `_extract_json`)
- 流式推理模式下每收到token检查 shutdown_event 支持中断
