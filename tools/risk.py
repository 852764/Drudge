"""Tool risk metadata used by the host-side approval boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


@dataclass(frozen=True, slots=True)
class ToolRisk:
    level: RiskLevel
    reason: str
    action: str

    @property
    def requires_approval(self) -> bool:
        return _RISK_ORDER[self.level] >= _RISK_ORDER[RiskLevel.MEDIUM]


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    tool_name: str
    arguments: dict
    risk: ToolRisk


class ApprovalDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


def coerce_risk_level(value: RiskLevel | str) -> RiskLevel:
    if isinstance(value, RiskLevel):
        return value
    try:
        return RiskLevel(str(value).lower())
    except ValueError as exc:
        raise ValueError(f"Unknown tool risk level: {value}") from exc
