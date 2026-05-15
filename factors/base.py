from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class FactorResult:
    factor_name: str
    score: int = 0                # 已废弃，LLM 全权分析
    signal: str = "llm"           # 已废弃
    detail: dict = field(default_factory=dict)   # 组织好的原始数据 {标签: 值}
    events: list[dict] = field(default_factory=list)  # 关键事件 [{title, sentiment, impact}]
    has_data: bool = True         # False = 所有数据源均无数据，该因子应被排除


class FactorBase(ABC):
    """因子基类。所有因子模块必须继承并实现 analyze()。"""

    name: str = "base"

    @abstractmethod
    async def analyze(self, code: str, name: str = "") -> FactorResult:
        ...
