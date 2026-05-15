import threading
import time
from dataclasses import dataclass, field
from typing import Optional

STAGES = [
    {"id": "fundamental", "name": "基本面", "icon": "📊"},
    {"id": "industry",    "name": "行业",   "icon": "🏭"},
    {"id": "macro",       "name": "宏观",   "icon": "🌍"},
    {"id": "fund_flow",   "name": "资金流向", "icon": "💰"},
    {"id": "sentiment",   "name": "情绪技术", "icon": "📈"},
    {"id": "geo_external","name": "地缘",   "icon": "🌐"},
    {"id": "limit",       "name": "涨跌停",  "icon": "🔒"},
    {"id": "llm_news",    "name": "LLM 新闻分析", "icon": "📰"},
    {"id": "llm_factors", "name": "LLM 因子分析", "icon": "🧠"},
    {"id": "llm_debate",  "name": "LLM 辩论分析", "icon": "⚖️"},
]


@dataclass
class ProgressTracker:
    code: str
    start_time: float = field(default_factory=time.time)
    stage_status: dict[str, str] = field(default_factory=dict)  # id → pending|active|done
    stage_reports: dict[str, str] = field(default_factory=dict)
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    is_complete: bool = False
    error: Optional[str] = None
    result: Optional[dict] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        for s in STAGES:
            self.stage_status[s["id"]] = "pending"

    def mark_stage(self, stage_id: str, status: str):
        with self._lock:
            if stage_id in self.stage_status:
                self.stage_status[stage_id] = status

    def mark_stage_report(self, stage_id: str, text: str):
        with self._lock:
            if stage_id in self.stage_status:
                self.stage_reports[stage_id] = text

    def add_tokens(self, tokens_in: int, tokens_out: int):
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out

    def mark_complete(self, result: dict | None = None):
        with self._lock:
            self.is_complete = True
            if result:
                self.result = result

    def mark_error(self, error: str):
        with self._lock:
            self.is_complete = True
            self.error = error

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        with self._lock:
            completed = sum(1 for v in self.stage_status.values() if v == "done")
            total = len(STAGES)
            current = next((k for k, v in self.stage_status.items() if v == "active"), "")
            return {
                "code": self.code,
                "elapsed": round(self.elapsed, 1),
                "completed": completed,
                "total": total,
                "pct": int(completed / total * 100) if total else 0,
                "current_stage": current,
                "stages": [
                    {**s, "status": self.stage_status.get(s["id"], "pending")}
                    for s in STAGES
                ],
                "llm_calls": self.llm_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "is_complete": self.is_complete,
                "error": self.error,
                "stage_reports": dict(self.stage_reports),
            }
