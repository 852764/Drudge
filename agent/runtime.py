"""Long-lived Agent runtime for interactive sessions."""

from __future__ import annotations

import asyncio
from typing import Any

from .drudge_agent import Agent


class AgentRuntime:
    """Own one Agent lifecycle and serialize turns on one event loop."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._started = False
        self._closed = False
        self._turn_lock = asyncio.Lock()
        self._active_turn: asyncio.Task | None = None

    @property
    def started(self) -> bool:
        return self._started and not self._closed

    @property
    def active(self) -> bool:
        return bool(self._active_turn and not self._active_turn.done())

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("AgentRuntime is closed")
        if self._started:
            return
        await self.agent.start()
        self._started = True

    async def run_turn(
        self,
        prompt: str,
        *,
        memory_entries: list[str] | None = None,
        skills: list[str] | None = None,
        stream_callback: Any = None,
    ) -> str:
        async with self._turn_lock:
            if self._closed:
                raise RuntimeError("AgentRuntime is closed")
            await self.start()
            self._active_turn = asyncio.current_task()
            try:
                return await self.agent.run(
                    prompt,
                    memory_entries=memory_entries,
                    skills=skills,
                    stream_callback=stream_callback,
                )
            finally:
                self._active_turn = None

    def cancel(self) -> None:
        self.agent.cancel()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        active = self._active_turn
        if active and active is not asyncio.current_task() and not active.done():
            self.agent.cancel()
            await asyncio.gather(active, return_exceptions=True)
        await self.agent.close()
        self._started = False

    async def __aenter__(self) -> "AgentRuntime":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
