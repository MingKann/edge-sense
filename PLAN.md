# Edge-Sense 后续计划

> 当前主分支: `master` (2026-05-29)
> 项目地址: https://github.com/MingKann/edge-sense

---

## P0 — 保持项目健康

- [x] **修复 CI 测试工作流** ✅ (2026-05-29)
  - libgl1 修复已生效，CI 全绿
  - 59 个测试全部通过 (46 原有 + 13 新增集成测试)

## P1 — 下一轮优化

- [x] **集成测试（模拟摄像头帧输入）** ✅ (2026-05-29)
  - `tests/test_integration.py` — 13 个测试覆盖全管线
  - 合成帧模拟: 静态/噪声/运动/LED/OCR 场景
  - 无需物理摄像头，CI 中可运行

- [x] **Docker 镜像瘦身** ✅ (2026-05-29)
  - 多阶段构建: builder 阶段编译 → runtime 阶段仅复制产物
  - `opencv-python-headless` 替代 `opencv-python`（省 ~100MB）
  - `requirements-prod.txt` 去除 pytest/pytest-benchmark
  - 预计镜像从 ~1GB+ 减到 ~400MB

## P2 — 功能增强

- [x] **前端增强：历史记录持久化** ✅ (2026-05-29)
  - `src/history.py` — SQLite 诊断存储（DiagnosisStore）
  - API: `GET /api/history`, `/api/history/stats`, `/api/history/trend`
  - 前端: Canvas 置信度趋势图（最近 50 条，30 秒自动刷新）

- [x] **自适应推理频率** ✅ (2026-05-29)
  - 静态场景 (none) → 每 60 帧推理一次
  - 轻微运动 (low) → 每 30 帧
  - 中/高强度运动 (medium/high) → 每 15 帧
  - 滞后机制 (hysteresis=3) 防振荡
  - `config.yaml` 中可调参数

## 备注

- Docker Desktop 不装，CI 自动构建覆盖发布需求
- 本地开发直接用 `python src/server.py`
- 新文件: `tests/test_integration.py`, `src/history.py`, `requirements-prod.txt`
- 修改: `Dockerfile`, `src/server.py`, `src/config.yaml`, `web/index.html`, `.gitignore`, `.github/workflows/docker.yml`
