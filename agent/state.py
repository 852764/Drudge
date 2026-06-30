"""Explicit state and trace events for an Agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any


class RunStatus(str, Enum):
    IDLE = "idle"
    WAITING_FOR_MODEL = "waiting_for_model"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    EXECUTING_TOOLS = "executing_tools"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_TURNS = "max_turns"


@dataclass(frozen=True, slots=True)
class RunEvent:
    status: RunStatus
    turn: int
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)


@dataclass(slots=True)
class AgentRunState:
    status: RunStatus = RunStatus.IDLE
    turn: int = 0
    error: str | None = None
    events: list[RunEvent] = field(default_factory=list)

    def transition(self, status: RunStatus, *, turn: int, **detail: Any) -> None:
        self.status = status
        self.turn = turn
        self.error = detail.get("error") if status is RunStatus.FAILED else None
        self.events.append(RunEvent(status=status, turn=turn, detail=detail))
