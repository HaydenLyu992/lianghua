from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class FactorResult:
    factor_name: str
    score: int                    # 0-100
    signal: str                   # bullish / bearish / neutral
    detail: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)  # 关键事件 [{title, sentiment, impact}]


class FactorBase(ABC):
    """因子基类。所有因子模块必须继承并实现 analyze()。"""

    name: str = "base"

    @abstractmethod
    async def analyze(self, code: str, name: str = "") -> FactorResult:
        ...
