"""核心 牛马，其他的牛马都是这个牛马来的 — 工具调用 + LLM 交互 + 上下文管理"""

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from .llm import LLMClient, create_client
from .output_filter import (
    DegenerateReasoningError,
    ReasoningTagFilter,
    filter_reasoning_text,
    sanitize_provider_items,
)
from .context_manager import (
    build_compacted_messages,
    build_context_summary_messages,
    build_repo_map,
    partition_messages_for_compaction,
    summarize_messages,
)
from .refusal import build_refusal_review_messages, is_refusal
from .storage import ConversationStore
from .project_instructions import load_project_instructions, render_project_instructions
from .skills import Skill, SkillManager
from .state import AgentRunState, RunStatus
from tools import (
    ApprovalDecision,
    ApprovalRequest,
    RiskLevel,
    ToolContext,
    ToolResult,
    registry,
)
from prompt import build_system_prompt
from config import get_config, ConfigManager
from utils import truncate_string

# 工具调用结果的最大字符数
MAX_TOOL_RESULT_CHARS = 10000


class Agent:
    """核心 Agent"""

    def __init__(
        self,
        config: ConfigManager | None = None,
        approval_callback: Callable[
            [ApprovalRequest], ApprovalDecision | str | Awaitable[ApprovalDecision | str]
        ] | None = None,
    ):
        self.config = config or get_config()
        self.llm: LLMClient | None = None
        self.utility_llm: LLMClient | None = None
        self.refusal_llm: LLMClient | None = None
        self._messages: list[dict] = []
        self._turn_count: int = 0
        self._total_tokens: int = 0
        self._utility_tokens: int = 0
        self.session_id: str | None = None
        self.store: ConversationStore | None = None
        self.tool_context: ToolContext | None = None
        self.run_state = AgentRunState()
        self.approval_callback = approval_callback
        self._session_approvals: set[tuple[str, str]] = set()
        self._cancel_event = asyncio.Event()
        self._active_task: asyncio.Task | None = None
        self._last_compaction: dict[str, Any] | None = None
        workspace = self.config.get("security", "workspace_root", default=".")
        self.skill_manager = SkillManager(
            workspace,
            max_chars=int(self.config.get("agent", "skill_max_chars", default=32_000)),
        )
        self.active_skill_names: list[str] = []
        self._init_store()

    def cancel(self) -> None:
        """Request cancellation of the current model/tool operation."""
        self._cancel_event.set()
        task = self._active_task
        if task and not task.done():
            task.get_loop().call_soon_threadsafe(task.cancel)

    async def _emit(self, callback: Any, text: str) -> None:
        if not callback or not text:
            return
        result = callback(text)
        if inspect.isawaitable(result):
            await result

    async def _call_model_filtered(
        self,
        tool_schemas: list[dict],
        stream_callback: Any,
    ) -> tuple[dict, str | None, list[dict], bool, bool, int]:
        hide_reasoning = bool(
            self.config.get("display", "hide_reasoning_tags", default=True)
        )
        max_reasoning_chars = int(
            self.config.get("agent", "reasoning_tag_max_chars", default=12_000)
        )
        recovery_attempts = max(
            0,
            int(self.config.get("agent", "reasoning_recovery_attempts", default=1)),
        )
        total_tokens = 0
        last_result: tuple[dict, str | None, list[dict], bool, bool, int] | None = None

        for attempt in range(recovery_attempts + 1):
            parser = ReasoningTagFilter(max_reasoning_chars=max_reasoning_chars)
            turn_streamed = False

            async def emit_delta(delta: str) -> None:
                nonlocal turn_streamed
                visible = parser.feed(delta) if hide_reasoning else delta
                if not visible:
                    return
                turn_streamed = True
                await self._emit(stream_callback, visible)

            request_messages = (
                self._reasoning_recovery_messages(self._messages)
                if attempt > 0
                else self._messages
            )
            try:
                response = await self.llm.chat(
                    request_messages,
                    tools=tool_schemas if tool_schemas else None,
                    stream_callback=emit_delta if stream_callback else None,
                    cancel_event=self._cancel_event,
                )
                if stream_callback and hide_reasoning:
                    tail = parser.finish()
                    if tail:
                        turn_streamed = True
                        await self._emit(stream_callback, tail)
            except DegenerateReasoningError:
                if attempt < recovery_attempts and not "".join(parser.visible_parts).strip():
                    await self._compress_context()
                    continue
                raise

            usage = response.get("usage", {})
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            raw_text = self.llm.extract_text(response)
            filtered = (
                filter_reasoning_text(
                    raw_text or "",
                    max_reasoning_chars=max_reasoning_chars,
                )
                if hide_reasoning and raw_text
                else None
            )
            text = filtered.text if filtered is not None else raw_text
            tool_calls = self.llm.extract_tool_calls(response)
            if hide_reasoning and response.get("provider_items"):
                response["provider_items"] = sanitize_provider_items(
                    response["provider_items"],
                    max_reasoning_chars=max_reasoning_chars,
                )
            reasoning_only = bool(
                filtered
                and filtered.saw_tag
                and not (text or "").strip()
                and not tool_calls
            )
            last_result = (
                response,
                text,
                tool_calls,
                turn_streamed,
                reasoning_only,
                total_tokens,
            )
            if reasoning_only and attempt < recovery_attempts:
                await self._compress_context()
                continue
            return last_result

        if last_result is not None:
            return last_result
        raise RuntimeError("Model reasoning recovery failed without a response")

    @staticmethod
    def _reasoning_recovery_messages(messages: list[dict]) -> list[dict]:
        requirement = (
            "OUTPUT REQUIREMENT: Give the final answer directly. Do not emit <think> tags, "
            "hidden chain-of-thought, or a reasoning transcript. Keep internal reasoning private."
        )
        recovered = [dict(message) for message in messages]
        for index, message in enumerate(recovered):
            if message.get("role") == "system":
                content = str(message.get("content") or "")
                recovered[index] = {**message, "content": f"{content}\n\n{requirement}"}
                return recovered
        return [{"role": "system", "content": requirement}] + recovered

    async def _approve_tool(self, tool_name: str, args: dict) -> bool:
        if not self.tool_context or self.tool_context.approval_mode != "on_request":
            return True
        risk = registry.assess_risk(tool_name, args, self.tool_context)
        if not risk.requires_approval:
            return True
        if risk.level is RiskLevel.CRITICAL:
            return False
        approval_key = (tool_name, risk.level.value)
        if approval_key in self._session_approvals:
            return True
        if self.approval_callback is None:
            return False

        self.run_state.transition(
            RunStatus.WAITING_FOR_APPROVAL,
            turn=self._turn_count,
            tool=tool_name,
            risk=risk.level.value,
            action=risk.action,
        )
        decision = self.approval_callback(ApprovalRequest(tool_name, args, risk))
        if inspect.isawaitable(decision):
            decision = await decision
        try:
            parsed = decision if isinstance(decision, ApprovalDecision) else ApprovalDecision(str(decision))
        except ValueError:
            parsed = ApprovalDecision.DENY
        if parsed is ApprovalDecision.ALLOW_SESSION:
            self._session_approvals.add(approval_key)
            return True
        return parsed is ApprovalDecision.ALLOW_ONCE

    def _init_store(self) -> None:
        storage_config = self.config.get_storage_config()
        if not storage_config.get("enabled", True):
            return
        self.store = ConversationStore(storage_config.get("path", "~/.drudge/drudge.db"))

    def list_skills(self) -> list[dict[str, Any]]:
        discovered = self.skill_manager.discover()
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "path": str(skill.path),
                "active": skill.name in self.active_skill_names,
            }
            for skill in discovered.values()
        ]

    def get_skill(self, name: str) -> Skill:
        return self.skill_manager.get(name)

    def activate_skill(self, name: str) -> Skill:
        skill = self.skill_manager.get(name)
        if skill.name not in self.active_skill_names:
            self.active_skill_names.append(skill.name)
        self._persist_session_extensions()
        self._refresh_system_message()
        return skill

    def deactivate_skill(self, name: str) -> bool:
        if name not in self.active_skill_names:
            return False
        self.active_skill_names.remove(name)
        self._persist_session_extensions()
        self._refresh_system_message()
        return True

    def clear_skills(self) -> None:
        self.active_skill_names.clear()
        self._persist_session_extensions()
        self._refresh_system_message()

    def _persist_session_extensions(self) -> None:
        if self.store and self.session_id:
            self.store.update_session_metadata(
                self.session_id,
                {"active_skills": list(self.active_skill_names)},
            )

    def new_session(self, *, clear_skills: bool = False) -> None:
        self._messages = []
        self.session_id = None
        self._turn_count = 0
        self._total_tokens = 0
        self._utility_tokens = 0
        self._last_compaction = None
        self._session_approvals.clear()
        self.run_state = AgentRunState()
        if clear_skills:
            self.active_skill_names.clear()

    def resume_session(self, session_id: str) -> dict[str, Any]:
        if not self.store:
            raise RuntimeError("Conversation storage is disabled")
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        rows = self.store.get_messages(session_id, limit=None)
        if not rows:
            raise RuntimeError(f"Session has no messages: {session_id}")

        metadata = session.get("metadata") or {}
        available = self.skill_manager.discover()
        self.active_skill_names = [
            str(name)
            for name in metadata.get("active_skills", [])
            if str(name) in available
        ]
        self.session_id = session_id
        self._messages = [self._restore_message(row) for row in rows]
        self._sanitize_historical_reasoning()
        assistant_turns = sum(1 for message in self._messages if message.get("role") == "assistant")
        self._turn_count = max(self.store.get_max_turn(session_id), assistant_turns)
        self._total_tokens = 0
        self._utility_tokens = 0
        self._last_compaction = None
        self._session_approvals.clear()
        repaired = self._repair_incomplete_tool_transactions()
        self._refresh_system_message()
        result = dict(session)
        result["message_count"] = len(self._messages)
        result["repaired_tool_calls"] = repaired
        result["active_skills"] = list(self.active_skill_names)
        return result

    def _sanitize_historical_reasoning(self) -> None:
        if not self.config.get("display", "hide_reasoning_tags", default=True):
            return
        max_chars = int(
            self.config.get("agent", "reasoning_tag_max_chars", default=12_000)
        )
        for message in self._messages:
            if message.get("role") != "assistant":
                continue
            if isinstance(message.get("content"), str):
                message["content"] = filter_reasoning_text(
                    message["content"],
                    max_reasoning_chars=max_chars,
                ).text
            if message.get("provider_items"):
                message["provider_items"] = sanitize_provider_items(
                    message["provider_items"],
                    max_reasoning_chars=max_chars,
                )

    @staticmethod
    def _restore_message(row: dict[str, Any]) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": row.get("role", "user"),
            "content": row.get("content"),
        }
        if row.get("tool_call_id"):
            message["tool_call_id"] = row["tool_call_id"]
        metadata = row.get("metadata") or {}
        for key in ("tool_calls", "provider_items"):
            if metadata.get(key):
                message[key] = metadata[key]
        return message

    def _repair_incomplete_tool_transactions(self) -> int:
        pending: dict[str, str] = {}
        for message in self._messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls", []) or []:
                    call_id = str(call.get("id") or "")
                    if call_id:
                        name = str((call.get("function") or {}).get("name") or "unknown")
                        pending[call_id] = name
            elif message.get("role") == "tool":
                pending.pop(str(message.get("tool_call_id") or ""), None)

        for call_id, name in pending.items():
            content = ToolResult.failure(
                f"Tool call interrupted before completion: {name}",
                interrupted=True,
            ).to_json()
            message = {"role": "tool", "tool_call_id": call_id, "content": content}
            self._messages.append(message)
            self._persist_message("tool", content, tool_call_id=call_id, metadata={"repaired": True})
        return len(pending)

    def _init_llm(self) -> None:
        """初始化 LLM 客户端"""
        if self.llm is None:
            mc = self.config.get_model_config()
            self.llm = create_client(mc)
        security = self.config.get_security_config()
        self.tool_context = ToolContext.from_config(security, self.config.get_toolsets())

    def _init_refusal_llm(self) -> None:
        if self.refusal_llm is None:
            mc = dict(self.config.get_model_config())
            review_config = self.config.get("agent", "refusal_review_model", default=None)
            if isinstance(review_config, dict):
                mc.update(review_config)
            self.refusal_llm = create_client(mc)

    def _get_utility_llm(self) -> LLMClient:
        """Return the configured low-cost client, or reuse the primary client."""
        self._init_llm()
        if not self.config.has_utility_model():
            return self.llm
        if self.utility_llm is None:
            self.utility_llm = create_client(self.config.get_utility_model_config())
        return self.utility_llm

    def _build_initial_messages(
        self,
        user_prompt: str,
        memory_entries: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> list[dict]:
        """构建初始消息列表"""
        system_content = self._build_system_content(memory_entries, skills)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

    def _build_system_content(
        self,
        memory_entries: list[str] | None = None,
        extra_skills: list[str] | None = None,
    ) -> str:
        toolsets = self.config.get_toolsets()
        workspace = self.config.get("security", "workspace_root", default=".")
        repo_map = None
        if self.config.get("agent", "repo_map_enabled", default=True):
            repo_map = build_repo_map(
                self.config.get("security", "workspace_root", default="."),
                max_files=int(self.config.get("agent", "repo_map_max_files", default=80)),
            )
        project_instructions = None
        if self.config.get("agent", "instructions_enabled", default=True):
            items = load_project_instructions(
                workspace,
                cwd=Path.cwd(),
                filename=str(self.config.get("agent", "instructions_filename", default="AGENTS.md")),
                max_chars=int(self.config.get("agent", "instructions_max_chars", default=64_000)),
            )
            project_instructions = render_project_instructions(items, workspace) if items else None

        discovered = self.skill_manager.discover()
        catalog = [(skill.name, skill.description) for skill in discovered.values()]
        loaded_skills = []
        for name in self.active_skill_names:
            skill = discovered.get(name)
            if skill:
                loaded_skills.append(skill.render())
        loaded_skills.extend(extra_skills or [])
        return build_system_prompt(
            toolsets,
            memory_entries,
            loaded_skills,
            repo_map=repo_map,
            project_instructions=project_instructions,
            skill_catalog=catalog,
        )

    def _refresh_system_message(
        self,
        memory_entries: list[str] | None = None,
        extra_skills: list[str] | None = None,
    ) -> None:
        if not self._messages:
            return
        system = {"role": "system", "content": self._build_system_content(memory_entries, extra_skills)}
        for index, message in enumerate(self._messages):
            if message.get("role") == "system":
                self._messages[index] = system
                return
        self._messages.insert(0, system)

    def _ensure_session(self, prompt: str, memory_entries: list[str] | None, skills: list[str] | None) -> None:
        if self._messages:
            self._messages.append({"role": "user", "content": prompt})
            self._persist_message("user", prompt)
            return

        self._messages = self._build_initial_messages(prompt, memory_entries, skills)
        if not self.store:
            return

        model = self.config.get("model", "name", default="unknown")
        self.session_id = self.store.create_session(
            prompt,
            model,
            cwd=str(Path.cwd()),
            metadata={"active_skills": list(self.active_skill_names)},
        )
        for message in self._messages:
            self._persist_message(message["role"], message.get("content"))

    def _persist_message(
        self,
        role: str,
        content: str | None,
        *,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.store and self.session_id:
            self.store.append_message(
                self.session_id,
                role,
                content,
                tool_call_id=tool_call_id,
                metadata=metadata,
            )

    def _persist_tool_call(self, name: str, args: dict, result: str) -> None:
        if self.store and self.session_id:
            self.store.append_tool_call(self.session_id, self._turn_count, name, args, result)

    async def _review_refusal_if_needed(
        self,
        user_prompt: str,
        response_text: str,
        stream_callback: Any = None,
    ) -> str:
        agent_config = self.config.get_agent_config()
        if not agent_config.get("refusal_review_enabled", True):
            return response_text
        if not is_refusal(response_text):
            return response_text

        notice = agent_config.get(
            "refusal_review_notice",
            "[Drudge] Detected a possible refusal. Running a safe second-pass review...",
        )
        self._persist_message("system", notice, metadata={"event": "refusal_review_started"})

        self._init_refusal_llm()
        review_messages = build_refusal_review_messages(user_prompt, response_text)
        review_response = await self.refusal_llm.chat(review_messages)
        usage = review_response.get("usage", {})
        self._total_tokens += usage.get("total_tokens", 0)
        review_text = self.refusal_llm.extract_text(review_response) or response_text
        self._messages.append({"role": "assistant", "content": review_text})
        self._persist_message("assistant", review_text, metadata={"refusal_review": True})
        reviewed = f"{notice}\n\n{review_text}"
        if stream_callback:
            await self._emit(stream_callback, f"\n\n{reviewed}")
        return reviewed

    async def run(
        self,
        prompt: str,
        memory_entries: list[str] | None = None,
        skills: list[str] | None = None,
        stream_callback: Any = None,
    ) -> str:
        """
        执行 Agent 对话循环

        Args:
            prompt: 用户输入的提示词
            memory_entries: 记忆条目列表
            skills: 加载的技能文档列表
            stream_callback: 可选的回调函数，接收每一轮的文本增量

        Returns:
            最终响应文本
        """
        self._init_llm()
        self._cancel_event = asyncio.Event()
        self._active_task = asyncio.current_task()
        self._refresh_system_message(memory_entries, skills)
        self._ensure_session(prompt, memory_entries, skills)

        max_turns = self.config.get_agent_config().get("max_turns", 50)
        compression_threshold = self.config.get_agent_config().get("compression_threshold", 0.80)
        toolsets = self.config.get_toolsets()
        tool_schemas = registry.get_schemas(toolsets)

        self.run_state = AgentRunState()
        final_text = ""

        for request_turn in range(1, max_turns + 1):
            self._turn_count += 1
            self.run_state.transition(
                RunStatus.WAITING_FOR_MODEL,
                turn=request_turn,
                total_turn=self._turn_count,
            )

            try:
                # 上下文压缩和正常回答共享取消、失败状态处理。
                estimated = self.llm.estimate_tokens(self._messages)
                context_limit = self.config.get("model", "context_length")
                if estimated > context_limit * compression_threshold:
                    await self._compress_context()

                (
                    response,
                    text,
                    tool_calls,
                    turn_streamed,
                    reasoning_only,
                    attempt_tokens,
                ) = await self._call_model_filtered(
                    tool_schemas,
                    stream_callback,
                )
            except asyncio.CancelledError:
                self.run_state.transition(RunStatus.CANCELLED, turn=request_turn)
                self._active_task = None
                raise
            except DegenerateReasoningError as e:
                error_msg = f"Model reasoning stream was stopped: {e}"
                self.run_state.transition(
                    RunStatus.FAILED,
                    turn=request_turn,
                    error=error_msg,
                )
                final_text = error_msg
                break
            except Exception as e:
                error_msg = f"LLM call failed (turn {self._turn_count}): {e}"
                self.run_state.transition(
                    RunStatus.FAILED,
                    turn=request_turn,
                    error=error_msg,
                )
                final_text = error_msg
                break

            # 统计 token 使用
            self._total_tokens += attempt_tokens

            finish_reason = self.llm.extract_finish_reason(response)
            if reasoning_only:
                error_msg = (
                    "Model returned only <think> reasoning and no final answer, "
                    "including after the configured recovery attempts. "
                    "Try /compact, /new, or a shorter prompt."
                )
                self.run_state.transition(
                    RunStatus.FAILED,
                    turn=request_turn,
                    error=error_msg,
                )
                final_text = error_msg
                break
            if tool_calls:
                self.run_state.transition(
                    RunStatus.EXECUTING_TOOLS,
                    turn=request_turn,
                    tool_count=len(tool_calls),
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": text or "",
                    "tool_calls": tool_calls,
                }
                if response.get("provider_items"):
                    assistant_msg["provider_items"] = response["provider_items"]
                self._messages.append(assistant_msg)
                metadata = {"tool_calls": tool_calls}
                if response.get("provider_items"):
                    metadata["provider_items"] = response["provider_items"]
                self._persist_message("assistant", text, metadata=metadata)

                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args_str = func.get("arguments", "{}")

                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = None

                    # 执行工具（支持 async handler）
                    if tool_args is None:
                        tool_result = json.dumps({
                            "ok": False,
                            "content": "",
                            "error": "Tool arguments are not valid JSON",
                            "metadata": {},
                        })
                        persisted_args = {"_raw": tool_args_str}
                    else:
                        try:
                            approved = await self._approve_tool(tool_name, tool_args)
                        except asyncio.CancelledError:
                            self.run_state.transition(RunStatus.CANCELLED, turn=request_turn)
                            self._active_task = None
                            raise
                        if approved:
                            self.run_state.transition(
                                RunStatus.EXECUTING_TOOLS,
                                turn=request_turn,
                                tool=tool_name,
                            )
                            try:
                                tool_result = await registry.dispatch_async(
                                    tool_name,
                                    tool_args,
                                    context=self.tool_context,
                                    approved=True,
                                )
                            except asyncio.CancelledError:
                                self.run_state.transition(RunStatus.CANCELLED, turn=request_turn)
                                self._active_task = None
                                raise
                        else:
                            risk = registry.assess_risk(tool_name, tool_args, self.tool_context)
                            error = (
                                f"Critical-risk tool call blocked: {risk.reason}"
                                if risk.level is RiskLevel.CRITICAL
                                else f"User approval denied for {tool_name}"
                            )
                            tool_result = ToolResult.failure(
                                error,
                                blocked=True,
                                approval_required=True,
                                risk=risk.level.value,
                                action=risk.action,
                            ).to_json()
                        persisted_args = tool_args if isinstance(tool_args, dict) else {"_raw": tool_args}
                    tool_result = truncate_string(tool_result, MAX_TOOL_RESULT_CHARS)

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": tool_result,
                    }
                    self._messages.append(tool_msg)
                    self._persist_message("tool", tool_result, tool_call_id=tc.get("id", ""))
                    self._persist_tool_call(tool_name, persisted_args, tool_result)

                    # 日志（如果配置开启了）
                    if self.config.get("display", "show_tool_calls"):
                        self._log_tool_call(tool_name, persisted_args, tool_result)

                # 继续下一轮
                continue

            if text:
                self._messages.append({"role": "assistant", "content": text})
                self._persist_message("assistant", text)
                if stream_callback and not turn_streamed:
                    await self._emit(stream_callback, text)

            if text:
                final_text = text
                self.run_state.transition(
                    RunStatus.COMPLETED,
                    turn=request_turn,
                    finish_reason=finish_reason,
                )
                break
            error_msg = "Model returned neither text nor tool calls"
            self.run_state.transition(
                RunStatus.FAILED,
                turn=request_turn,
                error=error_msg,
            )
            final_text = error_msg
            break
        else:
            self.run_state.transition(RunStatus.MAX_TURNS, turn=max_turns)
            final_text = f"[Agent reached maximum turns ({max_turns}). Task may be incomplete.]"

        try:
            reviewed = await self._review_refusal_if_needed(prompt, final_text, stream_callback)
        except asyncio.CancelledError:
            self.run_state.transition(RunStatus.CANCELLED, turn=self.run_state.turn)
            self._active_task = None
            raise
        self._active_task = None
        return reviewed

    async def _compress_context(self) -> dict[str, Any]:
        """Compact old context with an LLM summary and a deterministic fallback."""
        keep_recent = int(self.config.get("agent", "compact_keep_recent", default=8))
        system_messages, old_messages, recent_messages = partition_messages_for_compaction(
            self._messages,
            keep_recent=keep_recent,
        )
        if not old_messages:
            result = {
                "mode": "not_needed",
                "summarized_messages": 0,
                "summary_tokens": 0,
            }
            self._last_compaction = result
            return result

        configured_mode = str(
            self.config.get("agent", "context_summary_mode", default="llm")
        ).strip().lower()
        fallback_enabled = bool(
            self.config.get("agent", "context_summary_fallback", default=True)
        )
        summary_tokens = 0
        summary_model: str | None = None
        error: str | None = None

        if configured_mode == "llm":
            try:
                summary_llm = self._get_utility_llm()
                summary_model = summary_llm.model
                response = await summary_llm.chat(
                    build_context_summary_messages(old_messages),
                    tools=None,
                    cancel_event=self._cancel_event,
                )
                usage = response.get("usage", {})
                summary_tokens = int(usage.get("total_tokens", 0) or 0)
                raw_summary = summary_llm.extract_text(response) or ""
                filtered = filter_reasoning_text(
                    raw_summary,
                    max_reasoning_chars=int(
                        self.config.get("agent", "reasoning_tag_max_chars", default=12_000)
                    ),
                )
                summary = filtered.text.strip()
                if not summary:
                    raise RuntimeError("context summary model returned no visible text")
                mode = "llm"
            except Exception as exc:
                if not fallback_enabled:
                    raise
                error = str(exc)
                summary = summarize_messages(old_messages)
                mode = "fallback"
        else:
            summary = summarize_messages(old_messages)
            mode = "fallback"

        self._total_tokens += summary_tokens
        self._utility_tokens += summary_tokens
        self._messages = build_compacted_messages(
            system_messages,
            summary,
            recent_messages,
        )
        result = {
            "mode": mode,
            "summarized_messages": len(old_messages),
            "summary_tokens": summary_tokens,
            "summary_model": summary_model,
        }
        if error:
            result["fallback_reason"] = error
        self._last_compaction = result
        return result

    async def compact_context(self) -> dict[str, Any]:
        before_messages = len(self._messages)
        before_tokens = LLMClient.estimate_tokens(self._messages)
        details = await self._compress_context()
        return {
            "before_messages": before_messages,
            "after_messages": len(self._messages),
            "before_tokens": before_tokens,
            "after_tokens": LLMClient.estimate_tokens(self._messages),
            **details,
        }

    def _log_tool_call(self, name: str, args: dict, result: str) -> None:
        """记录工具调用（彩色输出）"""
        try:
            from colorama import Fore, Style, init
            init()
            print(f"{Fore.CYAN}[TOOL] {name}{Style.RESET_ALL}")
            if args:
                arg_str = json.dumps(args, ensure_ascii=False)
                if len(arg_str) > 150:
                    arg_str = arg_str[:150] + "..."
                print(f"  args: {arg_str}")
            if result:
                preview = result[:200].replace("\n", " ")
                print(f"  result: {preview}")
            print()
        except ImportError:
            pass

    def get_token_usage(self) -> dict:
        """获取 token 使用统计"""
        return {
            "total_tokens": self._total_tokens,
            "utility_tokens": self._utility_tokens,
            "turns": self._turn_count,
        }

    def get_status(self) -> dict[str, Any]:
        estimated = LLMClient.estimate_tokens(self._messages)
        context_limit = int(self.config.get("model", "context_length", default=0) or 0)
        context_percent = (estimated / context_limit * 100) if context_limit else None
        storage = self.config.get_storage_config()
        return {
            "session_id": self.session_id,
            "run_status": self.run_state.status.value,
            "model": self.config.get("model", "name", default="unknown"),
            "provider": self.config.get("model", "provider", default="openai-compatible"),
            "model_api": self.config.get("model", "api", default="auto"),
            "turns": self._turn_count,
            "tokens_this_process": self._total_tokens,
            "utility_tokens_this_process": self._utility_tokens,
            "utility_model": self.config.get_utility_model_config().get("name"),
            "utility_model_configured": self.config.has_utility_model(),
            "message_count": len(self._messages),
            "estimated_context_tokens": estimated,
            "context_limit": context_limit,
            "context_used_percent": context_percent,
            "workspace": str(self.config.get("security", "workspace_root", default=".")),
            "approval_mode": self.config.get("security", "approval_mode", default="auto"),
            "active_skills": list(self.active_skill_names),
            "storage_enabled": bool(storage.get("enabled", True)),
            "storage_path": storage.get("path") if storage.get("enabled", True) else None,
            "last_compaction": dict(self._last_compaction) if self._last_compaction else None,
        }

    def get_messages(self) -> list[dict]:
        """获取当前消息历史"""
        return list(self._messages)

    def get_run_state(self) -> AgentRunState:
        return self.run_state
