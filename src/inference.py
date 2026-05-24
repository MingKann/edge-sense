"""
Ollama HTTP API 推理封装 v3 (阶段3)
- /api/generate 同步调用
- JSON模式 + 容错提取
- 双超时 (冷启动300s/热推理120s)
- warmup 预加载
- 模型永久驻留显存 (keep_alive=-1)
- 限制上下文窗口 (num_ctx=1024)
- 限制生成token数 (num_predict=128)
"""

import json
import re
import requests
from typing import Optional, Dict, Any


class OllamaInference:
    """Ollama HTTP API 封装（本地 localhost:11434）"""

    def __init__(
        self,
        model: str = "edge-sense",
        base_url: str = "http://localhost:11434",
        hot_timeout: int = 120,        # 热推理超时：模型已在显存中
        cold_timeout: int = 300,       # 冷启动超时：首次加载2GB到显存
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.hot_timeout = hot_timeout
        self.cold_timeout = cold_timeout
        self._first_call = True        # 标记是否首次推理（冷启动）

    # ── 公开方法 ─────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str | Dict[str, Any]:
        """
        发送 prompt 到 Ollama，返回文本或解析后的JSON dict

        参数:
            prompt:      用户提示词
            system:      系统级提示词（角色设定）
            temperature: 生成温度（阶段3用0.2以稳定JSON输出）
            json_mode:   若True，payload加入 format:"json"，且返回解析后的dict
        返回:
            若 json_mode=False → str
            若 json_mode=True  → dict（解析后的JSON对象）
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": -1,           # 模型永久驻留显存，避免每次推理重新加载
            "options": {
                "temperature": temperature,
                "num_predict": 128,     # JSON诊断≤128token，减少冗余生成
                "num_ctx": 1024,        # 上下文窗口1024（诊断不需要长上下文）
            },
        }

        # R4 缓解：Ollama API 级 JSON 约束
        if json_mode:
            payload["format"] = "json"

        # 双超时选择
        timeout = self.cold_timeout if self._first_call else self.hot_timeout

        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # 首次调用完成 → 后续使用热超时
        if self._first_call:
            self._first_call = False

        raw_text = data.get("response", "").strip()

        # R1 缓解：JSON模式时走容错提取管线
        if json_mode:
            return self._extract_json(raw_text)
        return raw_text

    def warmup(self) -> None:
        """
        预热模型：发送最小prompt触发Ollama加载模型到显存
        必须在摄像头循环之前调用，避免冷启动超时影响首帧推理
        """
        print("[warmup] 正在加载模型到显存...", end=" ", flush=True)
        self.generate("1+1=", temperature=0, json_mode=False)
        # 注意：warmup调用后 _first_call 已变为 False
        # 后续 generate() 全部使用 hot_timeout
        print("完成")

    def ping(self) -> bool:
        """检测 Ollama 服务是否在线"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # ── 内部方法 ─────────────────────────────────

    @staticmethod
    def _extract_json(raw_text: str) -> Dict[str, Any]:
        """
        从LLM输出中提取合法JSON对象（四层容错回退）

        3B量化模型常见JSON输出缺陷:
          - 末尾附加解释文字
          - 包裹在 ```json ... ``` 代码块中
          - 尾部多余逗号
          - 字段值中包含未转义引号（罕见但可能）

        四层策略:
          L1: 直接 json.loads（理想路径）
          L2: 去除markdown代码围栏后 json.loads
          L3: 正则提取 {...} 块后 json.loads
          L4: 修复尾部逗号后 json.loads
        """
        text = raw_text.strip()

        # L1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # L2: 去除 markdown 代码围栏 ```json ... ```
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # L3: 正则提取第一个 {...} 块（处理末尾附加文本）
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not match:
            raise ValueError(
                f"LLM输出中未找到JSON对象，原始输出前200字符:\n{text[:200]}"
            )
        candidate = match.group(0)

        # L4: 修复常见语法错误（尾部逗号）
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = re.sub(r',\s*}', '}', candidate)   # {"a":1,} → {"a":1}
            fixed = re.sub(r',\s*]', ']', fixed)       # [1,2,] → [1,2]
            return json.loads(fixed)                    # L4失败则直接抛异常
